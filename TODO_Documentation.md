# FloatChat MVP: Technical TODO & Documentation

This document outlines the technical implementation plan for the FloatChat MVP. The goal is to create a functional prototype of an AI-powered conversational interface for ARGO ocean data by tomorrow.

**Project Goal:** A web-based chatbot where users can ask questions in natural language about ARGO float data and receive answers in the form of text, tables, and visualizations.

**MVP Tech Stack:**
*   **Backend Logic:** Python
*   **LLM Framework:** LangChain
*   **LLM:** Google Gemini API
*   **Data Retrieval:** `argopy` library
*   **Frontend:** Streamlit
*   **Data Visualization:** Plotly
*   **Data Manipulation:** Pandas, Xarray

---

## Phase 1: Environment and Backend Setup

### Step 1.1: Setup Python Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create a conda environment
conda create -n floatchat python=3.9
conda activate floatchat

# Or use venv
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

### Step 1.2: Install Dependencies
Install all necessary libraries. `argopy` will pull `xarray` and `pandas`.

```bash
pip install langchain google-generativeai python-dotenv argopy streamlit plotly pandas
```

### Step 1.3: Google Gemini API Key
1.  Obtain your API key from Google AI Studio.
2.  Create a file named `.env` in the root of your project directory.
3.  Add your API key to the `.env` file:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
    **IMPORTANT:** Add `.env` to your `.gitignore` file to avoid committing your key.

### Step 1.4: Core Application File
Create a single Python file for the Streamlit application, e.g., `app.py`. This will contain the backend and frontend code for the MVP.

---

## Phase 2: Backend - LangChain Agent & Tools

The core of the application will be a LangChain agent that can intelligently use tools to answer user queries.

### Step 2.1: Define Custom Tools for `argopy`
In `app.py`, we will define functions that the LLM can call. These functions will wrap the `argopy` library's functionality.

**Tool 1: Fetch Data by Region**
*   **Purpose:** Get data for a specific geographical bounding box and time frame.
*   **Implementation:**
    ```python
    # In app.py
    from argopy import DataFetcher
    from langchain.tools import tool

    @tool
    def fetch_argo_data_by_region(box: list):
        """
        Fetches ARGO data for a given geographical region, depth, and date range.
        The input 'box' must be a list of floats with 8 values:
        [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_start, date_end].
        Example: [-85, -45, 10, 20, 0, 100, '2023-01-01', '2023-03-31']
        Returns the data as a pandas DataFrame.
        """
        try:
            ds = DataFetcher().region(box).to_xarray()
            return ds.to_dataframe()
        except Exception as e:
            return f"Error fetching data: {e}"
    ```

**Tool 2: Fetch Data by Float ID**
*   **Purpose:** Get data for one or more specific ARGO floats.
*   **Implementation:**
    ```python
    # In app.py
    @tool
    def fetch_argo_data_by_float(float_ids: list[int]):
        """
        Fetches ARGO data for a given list of float WMO numbers (IDs).
        Returns the data as a pandas DataFrame.
        """
        try:
            ds = DataFetcher().float(float_ids).to_xarray()
            return ds.to_dataframe()
        except Exception as e:
            return f"Error fetching data: {e}"
    ```

**Tool 3: Create Visualizations**
*   **Purpose:** Generate plots from the fetched data. The LLM will decide what to plot based on the user's request.
*   **Implementation:**
    ```python
    # In app.py
    import plotly.express as px
    import pandas as pd

    @tool
    def create_argo_visualization(data_json: str, plot_type: str, x_axis: str, y_axis: str, color_by: str = None):
        """
        Creates a visualization from ARGO data.
        'data_json': The data to plot, as a JSON string.
        'plot_type': Type of plot. Can be 'scatter', 'line', 'histogram'.
        'x_axis': The column name for the x-axis.
        'y_axis': The column name for the y-axis.
        'color_by': (Optional) The column name to use for color differentiation.
        Returns a Plotly figure object.
        """
        df = pd.read_json(data_json)
        if 'PRES' in df.columns: # Invert pressure axis for depth plots
            df['PRES_inverted'] = -df['PRES']
            if y_axis == 'PRES':
                y_axis = 'PRES_inverted'

        if plot_type == 'scatter':
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, title=f'{y_axis} vs. {x_axis}')
        elif plot_type == 'line':
            fig = px.line(df, x=x_axis, y=y_axis, color=color_by, title=f'{y_axis} vs. {x_axis}')
        else:
            return "Unsupported plot type. Use 'scatter' or 'line'."

        if y_axis == 'PRES_inverted':
             fig.update_layout(yaxis_title="Pressure (Depth)")

        return fig
    ```

### Step 2.2: Initialize the LangChain Agent
*   **Purpose:** Combine the LLM, the tools, and a prompt to create an agent that can reason and execute tasks.
*   **Implementation:**
    ```python
    # In app.py
    import os
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain import hub

    load_dotenv()

    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

    # 2. Define the list of tools
    tools = [fetch_argo_data_by_region, fetch_argo_data_by_float, create_argo_visualization]

    # 3. Get the agent prompt template
    prompt = hub.pull("hwchase17/react")

    # 4. Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    ```

---

## Phase 3: Frontend - Streamlit Chat Interface

### Step 3.1: Basic App Structure
*   **Purpose:** Set up the title and the chat message history.
*   **Implementation:**
    ```python
    # In app.py
    import streamlit as st

    st.title("FloatChat 🌊 - ARGO Ocean Data Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "figure" in message:
                st.plotly_chart(message["figure"])
    ```

### Step 3.2: Handle User Input
*   **Purpose:** Capture user input and pass it to the LangChain agent.
*   **Implementation:**
    ```python
    # In app.py
    if prompt := st.chat_input("Ask about ARGO float data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Invoke the agent
            response = agent_executor.invoke({"input": prompt})
            output_text = response.get("output", "I encountered an error.")

            # The agent's final output might be text, or it might be a figure from the visualization tool.
            # This part needs refinement. For the MVP, we can check the type of the last tool's output.
            # A more robust solution would involve a structured output agent.

            # Simple MVP logic:
            if isinstance(output_text, go.Figure): # Check if the output is a plotly figure
                 st.plotly_chart(output_text)
                 st.session_state.messages.append({"role": "assistant", "content": "Here is the visualization you requested:", "figure": output_text})
            else:
                 st.markdown(output_text)
                 st.session_state.messages.append({"role": "assistant", "content": output_text})
    ```
    *Note: The agent's output handling will need to be refined. The agent might output a plot object directly or text. The code above provides a basic starting point.*

---

## Phase 4: Running the Application

1.  Make sure your `.env` file is in the same directory as `app.py`.
2.  Open your terminal, activate your virtual environment (`conda activate floatchat`).
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4.  Your web browser should open with the FloatChat interface.

---

## Granular Task List & Team Distribution

-   **Task 1 (Backend Lead):**
    -   [ ] Set up the complete `app.py` file structure.
    -   [ ] Implement and test the `fetch_argo_data_by_region` tool.
    -   [ ] Implement and test the `fetch_argo_data_by_float` tool.
    -   [ ] Ensure the `.env` file is correctly loaded.

-   **Task 2 (Agent/LLM Lead):**
    -   [ ] Implement the `create_argo_visualization` tool.
    -   [ ] Initialize the Gemini LLM and the LangChain agent (`create_react_agent`).
    -   [ ] Create the `AgentExecutor`.
    -   [ ] **Crucial:** Refine the prompt template (`prompt`) to give the agent better instructions on how to use the tools and how to format its final answer. Tell it to think step-by-step and first fetch data, then decide if a plot is needed.

-   **Task 3 (Frontend Lead):**
    -   [ ] Set up the Streamlit chat history and message display loop.
    -   [ ] Implement the user input handling (`st.chat_input`).
    -   [ ] Integrate the call to `agent_executor.invoke()`.
    -   [ ] Implement the logic to display the agent's response, correctly handling text vs. Plotly figures.

### Example User Queries to Test:
*   "Show me temperature data for floats [6902746, 6902747]"
*   "Get data near the equator between longitude -80 and -50 and latitude -5 and 5 for January 2023"
*   "Fetch data for float 6902746 and plot temperature vs pressure"
*   "Compare the salinity profiles for floats 6902746 and 6902757 using a scatter plot with pressure on the y-axis and salinity on the x-axis"
