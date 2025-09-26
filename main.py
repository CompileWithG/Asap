# main.py - FastAPI application

import asyncio
import json
import os
import re
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware



from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import your existing components
from dotenv import load_dotenv
from argopy import ArgoIndex, DataFetcher
import plotly.express as px
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="FloatChat API",
    description="ARGO Ocean Data Assistant API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Next.js default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directory exists
OUTPUT_DIR = Path("out_img")
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files for serving images
app.mount("/images", StaticFiles(directory=str(OUTPUT_DIR)), name="images")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    images: List[str] = []
    agent_steps: List[Dict[str, Any]] = []

class AgentStep(BaseModel):
    step_type: str  # "thought", "action", "observation"
    content: str
    timestamp: datetime

class WebSocketMessage(BaseModel):
    type: str  # "agent_step", "final_response", "error"
    data: Any

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except:
                # Connection might be closed, remove it
                self.disconnect(session_id)

manager = ConnectionManager()

# Your existing helper functions
def parse_action_input(tool_input):
    """Parse tool input, handling JSON wrapped in markdown"""
    if isinstance(tool_input, dict):
        return tool_input

    s = str(tool_input).strip()
    
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
    """Parse BGC tool input to extract parameter"""
    if isinstance(tool_input, dict):
        if "parameter" in tool_input:
            return tool_input["parameter"]
        else:
            raise ValueError("Input dictionary is missing the 'parameter' field.")

    s = str(tool_input).strip()

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



# Complete tools implementation from your original app.py
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
    {"plot_opt": ["scatter","scatter_3d","scatter_polar","scatter_ternary","line","line_3d","line_polar","area","bar","histogram","violin","box","strip","pie","sunburst","treemap","icicle","funnel","funnel_area","density_contour","density_heatmap","scatter_geo","choropleth","choropleth_mapbox","scatter_mapbox","density_mapbox","parallel_coordinates","parallel_categories","imshow"]} - Plotting options for best visualization 
    {"x": "PRES", "y": "TEMP"} - Plotting parameters if plot is True
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

        query_params["date_start"] = parsed.get("date_start", "2023-01-01")
        query_params["date_end"] = parsed.get("date_end", "2024-12-31")

        box_numeric = [
            query_params["lon_min"], query_params["lon_max"],
            query_params["lat_min"], query_params["lat_max"],
            query_params["pres_min"], query_params["pres_max"],
            query_params["date_start"], query_params["date_end"]
        ]
        
        try:
            ds = DataFetcher(fs_opts={'timeout': 60}).region(box_numeric).to_xarray()
            df = ds.to_dataframe()
        except Exception as e:
            return f"Error fetching data: {e} most likely the amount of data we are fetching is too much so try to reduce the param which we are controlling"

        if df.empty:
            return "No data found for the specified region and time."
            
        filters_applied = []
        # Reset index (N_POINTS can be an index) and summarise
        df_reset = df.reset_index()
        
        # Apply all the filters from your original implementation
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
        
        # Complete plotting implementation
        if parsed.get("plot", False):
            plot_item = list(parsed.get("plot_opt", ["scatter"]))
            print(plot_item)
            plot_message = ""  
            os.makedirs("out_img", exist_ok=True)
            fig = None
            
            for plot in plot_item:
                try:
                    if plot == "scatter":
                        fig = px.scatter(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                       title=f"ARGO Data: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "scatter_3d":
                        z = parsed.get("z", "PRES")
                        fig = px.scatter_3d(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), z=z, 
                                          title=f"ARGO 3D: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')} vs. {z}")
                    elif plot == "scatter_polar":
                        fig = px.scatter_polar(data_frame=df_reset, r=parsed.get("x", "PRES"), theta=parsed.get("y", "TEMP"), 
                                             title=f"ARGO Polar: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "scatter_ternary":
                        a = parsed.get("a", parsed.get("x", "TEMP"))
                        b = parsed.get("b", parsed.get("y", "PSAL"))
                        c = parsed.get("c", "PRES")
                        fig = px.scatter_ternary(data_frame=df_reset, a=a, b=b, c=c, title=f"ARGO Ternary: {a}-{b}-{c}")
                    elif plot == "line":
                        fig = px.line(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                    title=f"ARGO Data: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "line_3d":
                        z = parsed.get("z", "PRES")
                        fig = px.line_3d(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), z=z, 
                                       title=f"ARGO 3D Line: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')} vs. {z}")
                    elif plot == "line_polar":
                        fig = px.line_polar(data_frame=df_reset, r=parsed.get("x", "PRES"), theta=parsed.get("y", "TEMP"), 
                                          title=f"ARGO Polar Line: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "area":
                        fig = px.area(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                    title=f"ARGO Area: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "bar":
                        fig = px.bar(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                   title=f"ARGO Bar: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "histogram":
                        fig = px.histogram(data_frame=df_reset, x=parsed.get("x", "PRES"), title=f"ARGO Histogram: {parsed.get('x', 'PRES')}")
                    elif plot == "violin":
                        fig = px.violin(data_frame=df_reset, y=parsed.get("y", "TEMP"), title=f"ARGO Violin: {parsed.get('y', 'TEMP')}")
                    elif plot == "box":
                        fig = px.box(data_frame=df_reset, y=parsed.get("y", "TEMP"), title=f"ARGO Box Plot: {parsed.get('y', 'TEMP')}")
                    elif plot == "strip":
                        fig = px.strip(data_frame=df_reset, y=parsed.get("y", "TEMP"), title=f"ARGO Strip Plot: {parsed.get('y', 'TEMP')}")
                    elif plot == "pie":
                        fig = px.pie(data_frame=df_reset, names=parsed.get("x", "PLATFORM_NUMBER"), values=parsed.get("y", "TEMP"), 
                                   title=f"ARGO Pie: {parsed.get('x', 'PLATFORM_NUMBER')}")
                    elif plot == "sunburst":
                        path = parsed.get("path", [parsed.get("x", "PLATFORM_NUMBER")])
                        fig = px.sunburst(data_frame=df_reset, path=path, values=parsed.get("y", "TEMP"), title=f"ARGO Sunburst")
                    elif plot == "treemap":
                        path = parsed.get("path", [parsed.get("x", "PLATFORM_NUMBER")])
                        fig = px.treemap(data_frame=df_reset, path=path, values=parsed.get("y", "TEMP"), title=f"ARGO Treemap")
                    elif plot == "icicle":
                        path = parsed.get("path", [parsed.get("x", "PLATFORM_NUMBER")])
                        fig = px.icicle(data_frame=df_reset, path=path, values=parsed.get("y", "TEMP"), title=f"ARGO Icicle")
                    elif plot == "funnel":
                        fig = px.funnel(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                      title=f"ARGO Funnel: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "funnel_area":
                        fig = px.funnel_area(data_frame=df_reset, names=parsed.get("x", "PLATFORM_NUMBER"), values=parsed.get("y", "TEMP"), 
                                           title=f"ARGO Funnel Area")
                    elif plot == "density_contour":
                        fig = px.density_contour(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                               title=f"ARGO Density Contour: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "density_heatmap":
                        fig = px.density_heatmap(data_frame=df_reset, x=parsed.get("x", "PRES"), y=parsed.get("y", "TEMP"), 
                                               title=f"ARGO Density Heatmap: {parsed.get('x', 'PRES')} vs. {parsed.get('y', 'TEMP')}")
                    elif plot == "scatter_geo":
                        if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                            fig = px.scatter_geo(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", 
                                               color=parsed.get("color", parsed.get("y", "TEMP")), title=f"ARGO Geographic Scatter")
                        else:
                            plot_message += f"❌ Geographic coordinates (LATITUDE, LONGITUDE) not available for scatter_geo. "
                            continue
                    elif plot == "choropleth":
                        locations = parsed.get("locations", "PLATFORM_NUMBER")
                        fig = px.choropleth(data_frame=df_reset, locations=locations, color=parsed.get("y", "TEMP"), 
                                          title=f"ARGO Choropleth")
                    elif plot == "choropleth_mapbox":
                        if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                            fig = px.choropleth_mapbox(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", 
                                                     color=parsed.get("y", "TEMP"), title=f"ARGO Choropleth Mapbox")
                        else:
                            plot_message += f"❌ Geographic coordinates not available for choropleth_mapbox. "
                            continue
                    elif plot == "scatter_mapbox":
                        if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                            fig = px.scatter_mapbox(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", 
                                                  color=parsed.get("color", parsed.get("y", "TEMP")), title=f"ARGO Scatter Mapbox")
                        else:
                            plot_message += f"❌ Geographic coordinates not available for scatter_mapbox. "
                            continue
                    elif plot == "density_mapbox":
                        if "LATITUDE" in df_reset.columns and "LONGITUDE" in df_reset.columns:
                            fig = px.density_mapbox(data_frame=df_reset, lat="LATITUDE", lon="LONGITUDE", 
                                                  title=f"ARGO Density Mapbox")
                        else:
                            plot_message += f"❌ Geographic coordinates not available for density_mapbox. "
                            continue
                    elif plot == "parallel_coordinates":
                        dimensions = parsed.get("dimensions", ["TEMP", "PSAL", "PRES"])
                        fig = px.parallel_coordinates(data_frame=df_reset, dimensions=dimensions, title=f"ARGO Parallel Coordinates")
                    elif plot == "parallel_categories":
                        dimensions = parsed.get("dimensions", [parsed.get("x", "PLATFORM_NUMBER"), parsed.get("y", "DATA_MODE")])
                        fig = px.parallel_categories(data_frame=df_reset, dimensions=dimensions, title=f"ARGO Parallel Categories")
                    elif plot == "imshow":
                        try:
                            pivot_data = df_reset.pivot_table(values=parsed.get("y", "TEMP"), 
                                                            index=parsed.get("x", "PRES"), 
                                                            columns=parsed.get("columns", "TIME"))
                            fig = px.imshow(pivot_data, title=f"ARGO Heatmap: {parsed.get('y', 'TEMP')}")
                        except Exception as e:
                            plot_message += f"❌ Cannot create imshow plot: {e}. Try using density_heatmap instead. "
                            continue
                    else:
                        plot_message += f"❌ Unable to find the {plot} option. "
                        continue

                    if fig:     
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"plot_{plot}_{parsed.get('x', 'PRES')}_{parsed.get('y', 'TEMP')}_{timestamp}.png"
                        image_path = os.path.join("out_img", filename)
                        fig.write_image(image_path)
                        plot_message += f"Plot {plot} saved to {filename}. "
                        
                except Exception as plot_error:
                    plot_message += f"❌ Error creating {plot} plot: {plot_error}. "
                    continue

        # Build comprehensive response
        response_parts = [
            f"✅ Found {final_count} ARGO profiles {date_info}"
        ]
        
        if filters_applied:
            response_parts.append(f"Filters: {', '.join(filters_applied)}")
        
        if param_stats:
            response_parts.append(' | '.join(param_stats))
            
        if plot_message:
            response_parts.append(f"Plots: {plot_message}")

        return ' | '.join(response_parts)
    
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
        x = idx.read_params(parameter)

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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bgc_map_{parameter}_{timestamp}.png"
        image_path = os.path.join("out_img", filename)
        fig.write_image(image_path)

        return f"✅ SUCCESS: Map generated and saved to {filename}. Found {len(df)} profiles with {parameter} data."
    except ValueError as ve:
        return f"Input parsing error: {ve}"
    except Exception as e:
        tb = traceback.format_exc()
        return f"Error generating map: {e}\nTraceback:\n{tb}"

# Initialize LLM and Agent - exact same as original
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
tools = [fetch_argo_data_by_region_plot, generate_bgc_parameter_map]

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

# Background task for file cleanup
async def cleanup_old_files():
    """Clean up image files older than 1 hour"""
    current_time = datetime.now().timestamp()
    for file_path in OUTPUT_DIR.glob("*.png"):
        if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
            try:
                file_path.unlink()
            except:
                pass







# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage, background_tasks: BackgroundTasks):
    """Process chat message and return response with any generated images"""
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        # Schedule cleanup task
        background_tasks.add_task(cleanup_old_files)
        
        # Execute agent
        response = agent_executor.invoke({"input": chat_message.message})
        response_text = response.get("output", "I encountered an error.")
        
        # Extract image filenames from response - improved pattern matching
        image_patterns = [
            r"saved to ([^\s,;]+\.png)", 
            r"saved to ([^\s,;]+\.jpg)",  
            r"Map generated and saved to ([^\s,;.]+\.png)",
            r"Plot [^s]+ saved to ([^\s,;.]+\.png)",
            r"([a-zA-Z_][a-zA-Z0-9_]*\.png)",  
            r"([a-zA-Z_][a-zA-Z0-9_]*\.jpg)",  
        ]
        
        images = []
        for pattern in image_patterns:
            matches = re.findall(pattern, response_text)
            for match in matches:
                # Handle comma-separated filenames
                if "," in match:
                    images.extend([img.strip() for img in match.split(",")])
                else:
                    images.append(match.strip())
        
        # Remove duplicates and filter existing files
        images = list(set(images))
        existing_images = [img for img in images if (OUTPUT_DIR / img).exists()]
        
        # Extract agent steps if available
        agent_steps = []
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step[0], step[1]
                    agent_steps.append({
                        "action": str(action),
                        "observation": str(observation),
                        "timestamp": datetime.now().isoformat()
                    })
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            images=existing_images,
            agent_steps=agent_steps
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Serve generated images"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename
    )

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time agent updates"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                try:
                    # Send initial acknowledgment
                    await manager.send_message(session_id, {
                        "type": "status",
                        "data": {"status": "processing", "message": "Agent is thinking..."}
                    })
                    
                    # Execute agent
                    response = agent_executor.invoke({"input": message_data.get("message", "")})
                    response_text = response.get("output", "I encountered an error.")
                    
                    # Extract images
                    images = []
                    image_patterns = [
                        r"saved to ([^\s,;]+\.png)",
                        r"Plot [^s]+ saved to ([^\s,;.]+\.png)",
                        r"Map generated and saved to ([^\s,;.]+\.png)",
                    ]
                    
                    for pattern in image_patterns:
                        matches = re.findall(pattern, response_text)
                        for match in matches:
                            if "," in match:
                                images.extend([img.strip() for img in match.split(",")])
                            else:
                                images.append(match.strip())
                    
                    existing_images = [img for img in images if (OUTPUT_DIR / img).exists()]
                    
                    # Send final response
                    await manager.send_message(session_id, {
                        "type": "final_response",
                        "data": {
                            "response": response_text,
                            "images": existing_images,
                            "session_id": session_id
                        }
                    })
                    
                except Exception as e:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "data": {"error": str(e)}
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(session_id)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )