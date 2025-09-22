# app.py
import streamlit as st
import pandas as pd
import os
import json
import re
import traceback
from dotenv import load_dotenv
import sys

#st.write(f"Python Executable: {sys.executable}")
#st.write(f"Sys Path: {sys.path}")

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
def fetch_argo_data_by_region(tool_input):
    """
    tool_input may be a string or dict. Expected fields:
      lon_min, lon_max, lat_min, lat_max,
      depth_min, depth_max, date_start, date_end
    """
    try:
        parsed = parse_action_input(tool_input)

        # Convert numeric fields
        lon_min = float(parsed["lon_min"])
        lon_max = float(parsed["lon_max"])
        lat_min = float(parsed["lat_min"])
        lat_max = float(parsed["lat_max"])
        depth_min = float(parsed["depth_min"])
        depth_max = float(parsed["depth_max"])
        date_start = parsed["date_start"]
        date_end = parsed["date_end"]

        box_numeric = [
            lon_min, lon_max, lat_min, lat_max,
            depth_min, depth_max, date_start, date_end
        ]

        ds = DataFetcher(fs_opts={'timeout': 60}).region(box_numeric).to_xarray()
        df = ds.to_dataframe()
        if df.empty:
            return "No data found for the specified region and time."

        # Reset index (N_POINTS can be an index) and summarise
        df_reset = df.reset_index()
        # some datasets may not have these columns - guard with get()
        if "PLATFORM_NUMBER" in df_reset.columns:
            num_profiles = len(df_reset["PLATFORM_NUMBER"].unique())
        else:
            num_profiles = df_reset.shape[0]

        # TIME may be pandas datetime dtype already
        if "TIME" in df_reset.columns:
            min_date = pd.to_datetime(df_reset["TIME"].min()).strftime("%Y-%m-%d")
            max_date = pd.to_datetime(df_reset["TIME"].max()).strftime("%Y-%m-%d")
        else:
            min_date = date_start
            max_date = date_end

        min_temp = df_reset["TEMP"].min() if "TEMP" in df_reset.columns else float("nan")
        max_temp = df_reset["TEMP"].max() if "TEMP" in df_reset.columns else float("nan")
        min_psal = df_reset["PSAL"].min() if "PSAL" in df_reset.columns else float("nan")
        max_psal = df_reset["PSAL"].max() if "PSAL" in df_reset.columns else float("nan")

        summary = (
            f"Successfully found data from {num_profiles} ARGO float profile(s) "
            f"between {min_date} and {max_date}.\n"
            f"Temperature ranges from {min_temp:.2f} to {max_temp:.2f} C.\n"
            f"Salinity ranges from {min_psal:.2f} to {max_psal:.2f}."
        )
        return summary
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error fetching data: {e}\nTraceback:\n{tb}"

@tool()
def generate_bgc_parameter_map(tool_input):
    """
    Generates a world map showing the data quality for a given BGC parameter.
    tool_input should be a JSON object with a "parameter" key, e.g., {"parameter": "DOXY"}.
    """
    try:
        parameter = parse_bgc_tool_input(tool_input)
        if not parameter:
            raise ValueError("Input is missing the 'parameter' field.")

        from argopy import ArgoIndex
        from argopy.plot import scatter_map
        import matplotlib.pyplot as plt

        # Load the BGC index
        idx = ArgoIndex(index_file='bgc-b').load()

        # Search for the parameter
        idx.search_param(parameter)
        if idx.N_MATCH == 0:
            return f"No data found for the BGC parameter: {parameter}"

        # Convert to DataFrame and extract data mode
        df = idx.to_dataframe()
        df["variables"] = df["parameters"].apply(lambda x: x.split())
        df[f"{parameter}_DM"] = df.apply(lambda x: x['parameter_data_mode'][x['variables'].index(parameter)] if parameter in x['variables'] else '', axis=1)

        # Generate the map
        fig, ax = scatter_map(df,
                                hue=f"{parameter}_DM",
                                cmap="data_mode",
                                figsize=(10, 6),
                                markersize=5)
        ax.set_title(f"Global Data Mode for BGC Parameter: {parameter}")

        # Save the map to a file
        image_path = "bgc_map.png"
        plt.savefig(image_path)
        plt.close(fig) # Close the figure to free up memory

        return f"Map generated and saved to {image_path}"
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error generating map: {e}\nTraceback:\n{tb}"

# --- LLM + Agent setup ---

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define the list of tools available to the agent
tools = [fetch_argo_data_by_region, generate_bgc_parameter_map]

# Improve the prompt to ask the agent to emit JSON for action inputs
react_prompt_template = '''Answer the following questions as best you can. Your final answer should be a comprehensive summary of your findings, incorporating the details from the observations you have made. You have access to the following tools:

{tools}

When you call a tool, the "Action Input" MUST be a valid JSON object corresponding to the tool's arguments.

If you are using `fetch_argo_data_by_region`, the JSON must contain these fields:
  {{
    "lon_min": <number>,
    "lon_max": <number>,
    "lat_min": <number>,
    "lat_max": <number>,
    "depth_min": <number>,
    "depth_max": <number>,
    "date_start": "YYYY-MM-DD",
    "date_end": "YYYY-MM-DD"
  }}

If you are using `generate_bgc_parameter_map`, the JSON must contain this field:
  {{
    "parameter": "The BGC parameter to visualize (e.g., 'DOXY', 'BBP700')"
  }}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (MUST be valid JSON as described)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question. Be descriptive and include details from your observations.

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(react_prompt_template)

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor which will run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


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
            st.image(message["image"])

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
            
            # Check if the response contains a path to a generated map
            if "Map generated and saved to" in response_text:
                image_path = response_text.split("Map generated and saved to")[-1].strip()
                st.markdown("Here is the map you requested:")
                st.image(image_path)
                st.session_state.messages.append({"role": "assistant", "content": "Here is the map you requested:", "image": image_path})
            else:
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
