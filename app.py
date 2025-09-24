# app.py
import streamlit as st
import pandas as pd
import os
import json
import re
import traceback
from dotenv import load_dotenv
import sys

from datetime import datetime 
from argopy import ArgoIndex
from argopy.plot import scatter_map
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from langchain.prompts import PromptTemplate

from argopy import DataFetcher

# Load environment variables from .env file
load_dotenv()

# --- Helper: parse action input robustly ---
def parse_action_input(tool_input):
    """
    Parses the tool_input, which is expected to be a JSON string,
    potentially wrapped in markdown code blocks.
    """
    if isinstance(tool_input, dict):
        return tool_input

    s = str(tool_input).strip()
    
    # Clean the string if it's wrapped in markdown json block
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from tool input: {s}. Error: {e}")

def parse_bgc_tool_input(tool_input):
    """
    Parses the tool_input for the BGC tool, which is expected to be a JSON string
    with a "parameter" key.
    """
    if isinstance(tool_input, dict):
        if "parameter" in tool_input:
            return tool_input["parameter"]
        else:
            raise ValueError("Input dictionary is missing the 'parameter' field.")

    s = str(tool_input).strip()

    # Clean the string if it's wrapped in markdown json block
    if s.startswith("```json"):
        s = s[7:]
    if s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "parameter" in obj:
            return obj["parameter"]
        else:
            raise ValueError("Parsed JSON is missing the 'parameter' field.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON from tool input: {s}. Error: {e}")

# --- Tools ---
@tool()
def fetch_argo_data_by_region_plot(tool_input):
    """
    Flexible ARGO data fetching tool that can handle:
    - Single coordinates (lat, lon) or bounding box (lat_min, lat_max, lon_min, lon_max)
    - Any ARGO parameters as filters: CYCLE_NUMBER, DATA_MODE, PLATFORM_NUMBER, etc.
    - Optional depth and time constraints
    - Parameter-based searches (TEMP, PSAL, PRES, etc.)
    
    Examples:
    {"lat": 15.5, "lon": 88.2} - Single point
    {"lat_min": 10, "lat_max": 20, "lon_min": 80, "lon_max": 90} - Bounding box
    {"platform_number": "2903334"} - Specific float
    {"temp_min": 20, "temp_max": 30} - Temperature range
    {"plot":True} or {"plot":False} - Whether to generate a plot if user asked or if possible for the given query then suggested
    {"plot_opt": "scatter"} or {"plot_opt": "line"} - Plotting options for best visualization
    {"x": "PRES", "y": "TEMP"} - Plotting parameters if plot is True

    Returns a image path.
    """
    try:
        parsed = parse_action_input(tool_input)
        print(f"Parsed input: {parsed}")
        
        argo_params = [
            "CYCLE_NUMBER", "DATA_MODE", "DIRECTION", "PLATFORM_NUMBER",
            "POSITION_QC", "PRES", "PRES_ERROR", "PRES_QC",
            "PSAL", "PSAL_ERROR", "PSAL_QC", 
            "TEMP", "TEMP_ERROR", "TEMP_QC",
            "TIME_QC", "LATITUDE", "LONGITUDE", "TIME"
        ]
        
        query_params = {}

        if "lat" in parsed and "lon" in parsed:
            # Single point - create small bounding box around it
            lat = float(parsed["lat"])
            lon = float(parsed["lon"])
            buffer = float(parsed.get("buffer", 1.0))  # Default 1 degree buffer
            query_params.update({
                "lat_min": lat - buffer,
                "lat_max": lat + buffer,
                "lon_min": lon - buffer,
                "lon_max": lon + buffer
            })
            print(f"Single point query: {lat}, {lon} with {buffer}° buffer")
            
        elif all(k in parsed for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            # Full bounding box
            query_params.update({
                "lat_min": float(parsed["lat_min"]),
                "lat_max": float(parsed["lat_max"]),
                "lon_min": float(parsed["lon_min"]),
                "lon_max": float(parsed["lon_max"])
            })
            print("Bounding box query")
            
        elif any(k in parsed for k in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            # Partial coordinates - fill defaults
            query_params.update({
                "lat_min": float(parsed.get("lat_min", -90)),
                "lat_max": float(parsed.get("lat_max", 90)),
                "lon_min": float(parsed.get("lon_min", -180)),
                "lon_max": float(parsed.get("lon_max", 180))
            })
            print("Partial coordinates - using global defaults for missing bounds")
        else:
            # No geographic constraints - global search
            query_params.update({
                "lat_min": -90, "lat_max": 90,
                "lon_min": -180, "lon_max": 180
            })
            print("Global search - no geographic constraints")

        query_params["pres_min"] = float(parsed.get("pres_min", 0))
        query_params["pres_max"] = float(parsed.get("pres_max", 2000))

        query_params["date_start"] = parsed.get("date_start", "2020-01-01")
        query_params["date_end"] = parsed.get("date_end", "2024-12-31")

        box_numeric = [
            query_params["lon_min"], query_params["lon_max"],
            query_params["lat_min"], query_params["lat_max"],
            query_params["pres_min"], query_params["pres_max"],
            query_params["date_start"], query_params["date_end"]
        ]

        ds = DataFetcher(fs_opts={'timeout': 60}).region(box_numeric).to_xarray()
        df = ds.to_dataframe()

        if df.empty:
            return "No data found for the specified region and time."

        # Reset index (N_POINTS can be an index) and summarise
        df_reset = df.reset_index()
        # some datasets may not have these columns - guard with get()
        if "platform_number" in parsed:
            platform = str(parsed["platform_number"])
            if "PLATFORM_NUMBER" in df_reset.columns:
                df_reset = df_reset[df_reset["PLATFORM_NUMBER"].astype(str) == platform]
                filters_applied.append(f"Platform {platform}")
        
        # Filter by cycle number
        if "cycle_number" in parsed:
            cycle = int(parsed["cycle_number"])
            if "CYCLE_NUMBER" in df_reset.columns:
                df_reset = df_reset[df_reset["CYCLE_NUMBER"] == cycle]
                filters_applied.append(f"Cycle {cycle}")
        
        # Filter by temperature range
        if "temp_min" in parsed or "temp_max" in parsed:
            if "TEMP" in df_reset.columns:
                temp_mask = pd.Series([True] * len(df_reset))
                if "temp_min" in parsed:
                    temp_min = float(parsed["temp_min"])
                    temp_mask &= (df_reset["TEMP"] >= temp_min)
                    filters_applied.append(f"Temp ≥ {temp_min}°C")
                if "temp_max" in parsed:
                    temp_max = float(parsed["temp_max"])
                    temp_mask &= (df_reset["TEMP"] <= temp_max)
                    filters_applied.append(f"Temp ≤ {temp_max}°C")
                df_reset = df_reset[temp_mask]
        
        # Filter by salinity range
        if "psal_min" in parsed or "psal_max" in parsed:
            if "PSAL" in df_reset.columns:
                psal_mask = pd.Series([True] * len(df_reset))
                if "psal_min" in parsed:
                    psal_min = float(parsed["psal_min"])
                    psal_mask &= (df_reset["PSAL"] >= psal_min)
                    filters_applied.append(f"Salinity ≥ {psal_min}")
                if "psal_max" in parsed:
                    psal_max = float(parsed["psal_max"])
                    psal_mask &= (df_reset["PSAL"] <= psal_max)
                    filters_applied.append(f"Salinity ≤ {psal_max}")
                df_reset = df_reset[psal_mask]
        
        # Filter by pressure/depth range
        if "pres_min" in parsed or "pres_max" in parsed:
            if "PRES" in df_reset.columns:
                pres_mask = pd.Series([True] * len(df_reset))
                if "pres_min" in parsed:
                    pres_min = float(parsed["pres_min"])
                    pres_mask &= (df_reset["PRES"] >= pres_min)
                    filters_applied.append(f"Pressure ≥ {pres_min} dbar")
                if "pres_max" in parsed:
                    pres_max = float(parsed["pres_max"])
                    pres_mask &= (df_reset["PRES"] <= pres_max)
                    filters_applied.append(f"Pressure ≤ {pres_max} dbar")
                df_reset = df_reset[pres_mask]
        
        # Filter by data mode
        if "data_mode" in parsed:
            data_mode = parsed["data_mode"].upper()
            if "DATA_MODE" in df_reset.columns:
                df_reset = df_reset[df_reset["DATA_MODE"] == data_mode]
                filters_applied.append(f"Data mode: {data_mode}")

        final_count = len(df_reset)

        if "TIME" in df_reset.columns:
            min_date = pd.to_datetime(df_reset["TIME"].min()).strftime("%Y-%m-%d")
            max_date = pd.to_datetime(df_reset["TIME"].max()).strftime("%Y-%m-%d")
            date_info = f"between {min_date} and {max_date}"
        else:
            date_info = f"in requested period {query_params['date_start']} to {query_params['date_end']}"

        # Parameter ranges
        param_stats = []
        if "TEMP" in df_reset.columns:
            temp_range = f"{df_reset['TEMP'].min():.2f} to {df_reset['TEMP'].max():.2f}°C"
            param_stats.append(f"🌡️ Temperature: {temp_range}")
        
        if "PSAL" in df_reset.columns:
            psal_range = f"{df_reset['PSAL'].min():.2f} to {df_reset['PSAL'].max():.2f}"
            param_stats.append(f"🧂 Salinity: {psal_range}")
            
        if "PRES" in df_reset.columns:
            pres_range = f"{df_reset['PRES'].min():.1f} to {df_reset['PRES'].max():.1f} dbar"
            param_stats.append(f"📊 Pressure: {pres_range}")
        
        if parsed["plot"]==True:
            if parsed["plot_opt"] == "scatter":
                fig = px.scatter(data_frame=df_reset, x=parsed["x"], y=parsed["y"],title=f"Agro Data: {parsed['x']} vs. {parsed['y']}")
            elif parsed["plot_opt"] == "line":
                fig = px.line(data_frame=df_reset, x=parsed["x"], y=parsed["y"],title=f"Agro Data: {parsed['x']} vs. {parsed['y']}")
            else:
                return "❌ Invalid kind. Choose 'scatter' or 'line'."

            os.makedirs("out_img", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plot_{parsed['plot_opt']}_{parsed['x']}_{parsed['y']}_{timestamp}.png"
            image_path = os.path.join("out_img", filename)
            fig.write_image(image_path)

            return f"✅ Plot generated and saved to {image_path}"
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error fetching data: {e}\nTraceback:\n{tb}"


   

@tool()
def generate_bgc_parameter_map(tool_input):
    """
    Create a global scatter map of data-quality modes for a given
    Bio-Geo-Chemical (BGC) parameter from the Argo float dataset.

    • Expects: JSON with a key "parameter" (e.g. {"parameter": "DOXY"})
    • Finds all float profiles containing that parameter.
    • Converts their data-mode flags (R/A/D) to numeric values.
    • Plots a world map colored by data-mode and saves it as a PNG.

    Useful for questions like:
    – "Show where dissolved oxygen data exists and its quality mode."
    – "Map nitrate observations and highlight validated profiles."
    """
    try:
        parameter = parse_bgc_tool_input(tool_input)
        if not parameter:
            raise ValueError("Input is missing the 'parameter' field.")

        idx = ArgoIndex(index_file='bgc-b').load()

        x=idx.read_params(parameter)

        if not x:
            return f"No data found for the BGC parameter: {parameter}"
        
        df = idx.to_dataframe() 

        df["variables"] = df["parameters"].apply(lambda x: x.split())
        df[f"{parameter}_DM"] = df.apply(lambda x: x['parameter_data_mode'][x['variables'].index(parameter)] if parameter in x['variables'] else '', axis=1)

        df['DM_num'] = df[f"{parameter}_DM"].map({'R':0,'A':1,'D':2})

        fig = px.scatter_geo(df, 
                            lat='latitude', lon='longitude',
                            color='DM_num',
                            color_continuous_scale='Viridis',
                            title=f"Global Data Mode for BGC Parameter: {parameter}")
        
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=f"bgc_map_{parameter}_{timestamp}.png"
        image_path = os.path.join("out_img", filename)
        fig.write_image(image_path)

        return f"✅ SUCCESS: Map generated and saved to {image_path}. Found {len(df)} profiles with {parameter} data."
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error generating map: {e}\nTraceback:\n{tb}"

# --- LLM + Agent setup ---

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Define the list of tools available to the agent
tools = [fetch_argo_data_by_region_plot, generate_bgc_parameter_map]

# Improve the prompt to ask the agent to emit JSON for action inputs
react_prompt_template = '''Answer the following questions as best you can. Your final answer should be a comprehensive summary of your findings, incorporating the details from the observations you have made. You have access to the following tools:

{tools}

When you call a tool, the "Action Input" MUST be a valid JSON object corresponding to the tool's arguments.

For `fetch_argo_data_by_region`, the JSON can contain ANY of these optional fields:
Geographic Options:
  - Single point: {{"lat": <number>, "lon": <number>, "buffer": <degrees>}}
  - Bounding box: {{"lat_min": <number>, "lat_max": <number>, "lon_min": <number>, "lon_max": <number>}}
  - Partial bounds: Any subset of lat_min, lat_max, lon_min, lon_max (others use global defaults)
  
Parameter Filters (all optional):
  - Depth/Pressure: "pres_min", "pres_max" (default: 0-2000 dbar)
  - Temperature: "temp_min", "temp_max" (in °C)
  - Salinity: "psal_min", "psal_max" (in PSU)
  - Time: "date_start", "date_end" (format: "YYYY-MM-DD", default: 2020-2024)
  - Platform: "platform_number" (specific ARGO float ID)
  - Cycle: "cycle_number" (specific measurement cycle)
  - Data Quality: "data_mode" ("R"=real-time, "A"=adjusted, "D"=delayed-mode)

Examples:
  - {{"lat": 15, "lon": 88}} - Single point near Bay of Bengal
  - {{"lat_min": 5, "lat_max": 25, "lon_min": 80, "lon_max": 100}} - Regional box
  - {{"temp_min": 28, "psal_max": 35}} - Global search with parameter filters
  - {{"platform_number": "2903334"}} - Specific ARGO float data
  - {{"lat": 10, "lon": 75, "temp_min": 25, "pres_max": 1000}} - Combined filters

For `generate_bgc_parameter_map`, the JSON must contain:
  {{"parameter": "BGC_PARAMETER_NAME"}}
  
Available BGC parameters: "DOXY" (oxygen), "BBP700" (backscattering), "BBP470", "CHLA" (chlorophyll), "NITRATE", "PH_IN_SITU_TOTAL", "DOWNWELLING_PAR"

IMPORTANT STOPPING CONDITIONS:
- If you receive a "✅ SUCCESS" observation from any tool, immediately proceed to "Final Answer"
- If a map/visualization is generated successfully, provide the image path in your Final Answer, do NOT repeat the same action
- If you get data from ARGO floats, analyze it and provide your Final Answer
- Do NOT call the same tool multiple times with the same parameters and get in a loop
- Use the most appropriate parameters based on the user's question

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (MUST be valid JSON as described)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Be descriptive and include details from your observations. If an image was generated, mention the file path.

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(react_prompt_template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor which will run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,max_iterations=3,early_stopping_method="force",handle_parsing_errors=True,return_intermediate_steps=True)


# --- Streamlit frontend ---

st.title("FloatChat 🌊 - ARGO Ocean Data Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            try:
                if os.path.exists(message["image"]):
                    st.image(message["image"])
                else:
                    st.warning(f"Image file not found: {message['image']}")
            except Exception as e:
                st.error(f"Error displaying image: {e}")

# Handle user input
if prompt_text := st.chat_input("Ask about ARGO float data..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Get assistant response by invoking the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": prompt_text})
            response_text = response.get("output", "I encountered an error.")
            
            # Display response text first
            st.markdown(response_text)
            
            # Better image path extraction - look for specific patterns
            image_path = None
            
            # Try multiple patterns to find the image path
            image_patterns = [
                r"saved to ([^\s,;]+\.png)", 
                r"saved to ([^\s,;]+\.jpg)",  
                r"saved to ([^\s,;]+\.html)", 
                r"Map generated and saved to ([^\s,;.]+\.png)",
                r"([a-zA-Z_][a-zA-Z0-9_]*\.png)",  
                r"([a-zA-Z_][a-zA-Z0-9_]*\.jpg)",  
            ]

            
            for pattern in image_patterns:
                match = re.search(pattern, response_text)
                if match:
                    image_path = match.group(1).strip()
                    break
            
            if image_path:
                # Simple path construction - images are always in out_img folder
                if not image_path.startswith("out_img"):
                    # If extracted path doesn't include out_img, add it
                    full_path = os.path.join("out_img", os.path.basename(image_path))
                else:
                    # If it already includes out_img, use as is
                    full_path = image_path
                
                if os.path.exists(full_path):
                    try:
                        if full_path.endswith('.png') or full_path.endswith('.jpg'):
                            st.image(full_path, caption="Generated Map", use_container_width=True)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text, 
                                "image": full_path
                            })
                        elif full_path.endswith('.html'):
                            with open(full_path, 'r') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=600)
                            st.success("🎉 Interactive map displayed successfully!")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_text + f"\n\nInteractive map: {full_path}"
                            })
                    except Exception as e:
                        st.error(f"❌ Error displaying image: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    st.write(f"❌ File not found: `{full_path}`")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
            else:
                st.write("ℹ️ No image path detected in response")
                st.session_state.messages.append({"role": "assistant", "content": response_text})