import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool

from argopy import DataFetcher

# --- 1. Backend Setup: Agent & Tools ---

# Load environment variables from .env file
load_dotenv()

@tool
def fetch_argo_data_by_region(box: list):
    """Fetches ARGO data for a given geographical region, depth, and date range.
    The input 'box' MUST be a list with exactly 8 string elements:
    [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_start, date_end].
    Depths are in meters (positive values). Dates should be in 'YYYY-MM-DD' format.
    Example for the Arabian Sea in Jan 2023: ['50', '78', '8', '25', '0', '100', '2023-01-01', '2023-01-31']
    Returns the data as a pandas DataFrame in string format, or an error message.
    """
    try:
        # Argopy expects numeric types for region, so we convert them
        box_numeric = [
            float(box[0]), float(box[1]), float(box[2]), float(box[3]),
            float(box[4]), float(box[5]), box[6], box[7]
        ]
        ds = DataFetcher().region(box_numeric).to_xarray()
        df = ds.to_dataframe()
        if df.empty:
            return "No data found for the specified region and time."
        # Return a string representation of the dataframe head for the agent to process
        return df.head().to_string()
    except Exception as e:
        return f"Error fetching data: {e}. Ensure the input format is correct."

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# Define the list of tools available to the agent
# For now, it's just the one tool.
tools = [fetch_argo_data_by_region]

# Get the prompt template for the ReAct agent
prompt = hub.pull("hwchase17/react")

# Create the agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor which will run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 2. Frontend Setup: Streamlit Chat Interface ---

st.title("FloatChat 🌊 - ARGO Ocean Data Assistant")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about ARGO float data..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response by invoking the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Invoke the agent with the user's prompt
            response = agent_executor.invoke({"input": prompt})
            response_text = response.get("output", "I encountered an error.")
            st.markdown(response_text)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
