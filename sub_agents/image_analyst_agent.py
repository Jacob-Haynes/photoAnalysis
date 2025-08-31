from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from google.adk.tools import agent_tool
from google.genai.types import GenerateContentConfig

from sub_agents.object_detection_agent import object_detector

load_dotenv()

# --- Configuration ---
GCP_MODEL = "gemini-2.5-flash"

# --- Agent Definition ---

image_analyst_agent = LlmAgent(
    name="image_analyst",
    description="Advanced image analysis agent that detects and segments prominent objects in images, providing detailed scene descriptions and visual analysis",
    model=GCP_MODEL,
    instruction="""You are an expert image analyst specializing in scene understanding and object detection. 

Your tasks:
1. Analyze the provided image to identify all prominent objects and features
2. Get bounding boxes of prominent objects in the scene using the object_detection_agent
3. Provide detailed descriptions of:
   - Main objects
   - Spatial relationships between objects
   - Scene context and setting
   - Colors, textures, and visual characteristics
   - Any notable features or anomalies

4. Return structured information including:
   - Comprehensive scene description
   - List of prominent objects with their properties
   - Segmentation results for objects
   - Bounding box coordinates for detected objects [ymin, xmin, ymax, xmax] normalized to 0-1000

Be thorough and descriptive in your analysis while maintaining accuracy.""",
    tools=[agent_tool.AgentTool(agent=object_detector)],
    output_key="analysis_output",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
)
