# app.py
import streamlit as st
import pandas as pd
import os
import json
import re
import traceback
from dotenv import load_dotenv

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
    Accepts tool_input which may be:
      - a dict (already structured)
      - a JSON string
      - a comma-separated 'key=value' string
      - 'key: value' or 'key value' forms
    Returns dict with keys or raises ValueError.
    """
    keys = [
        "lon_min", "lon_max", "lat_min", "lat_max",
        "depth_min", "depth_max", "date_start", "date_end"
    ]

    # If already dict-like, normalize keys and return
    if isinstance(tool_input, dict):
        return {k: str(tool_input.get(k)) for k in keys if k in tool_input}

    s = str(tool_input).strip()

    # 1) try strict JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {k: str(obj.get(k)) for k in keys if k in obj}
    except Exception:
        pass

    # 2) try to find patterns like key=val, key: val, "key": val, or key val
    result = {}
    for k in keys:
        # Try key = "value" or key: "value"
        m = re.search(rf'{re.escape(k)}\s*[:=]\s*["\']?([^\s,"\']+)["\']?', s, flags=re.IGNORECASE)
        if m:
            result[k] = m.group(1)
            continue
        # Try key <space> value (e.g., "date_start 2022-08-01")
        m2 = re.search(rf'{re.escape(k)}\s+([^\s,]+)', s, flags=re.IGNORECASE)
        if m2:
            result[k] = m2.group(1)
            continue

    # 3) fallback: split by commas and parse key=value pairs
    if not result:
        parts = re.split(r',|\n', s)
        for p in parts:
            if '=' in p:
                k, v = p.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k in keys:
                    result[k] = v

    # 4) final check: ensure we have all keys
    missing = [k for k in keys if k not in result]
    if missing:
        raise ValueError(f"Missing required fields in tool input: {missing}. Received raw input: {s}")

    return result

# --- The tool (no args_schema so the agent can send free-form input) ---
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

        ds = DataFetcher().region(box_numeric).to_xarray()
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

# --- LLM + Agent setup ---

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Define the list of tools available to the agent
tools = [fetch_argo_data_by_region]

# Improve the prompt to ask the agent to emit JSON for action inputs
react_prompt_template = '''Answer the following questions as best you can. Your final answer should be a comprehensive summary of your findings, incorporating the details from the observations you have made. You have access to the following tools:

{tools}

IMPORTANT: When you call a tool, the "Action Input" MUST be a valid JSON object with these exact fields:
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

Example of a valid Action Input:
Action Input: {{ "lon_min": -125, "lon_max": -117, "lat_min": 32, "lat_max": 42, "depth_min": 0, "depth_max": 2000, "date_start": "2022-08-01", "date_end": "2022-08-31" }}

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
            st.markdown(response_text)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
