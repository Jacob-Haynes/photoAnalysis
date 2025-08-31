from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

load_dotenv()

# --- Configuration ---
GCP_MODEL = "gemini-2.5-flash"

# --- Agent Definition ---

object_detector = LlmAgent(
    name="object_detector",
    description="Advanced image analysis agent that detects and segments notable objects providing visual analysis with bounding boxes",
    model=GCP_MODEL,
    instruction="""
  Identify the objects in this image.
  Output a JSON list where each entry contains the 2D bounding box 
  in the key "box_2d" and the text label in the key "label". 
  Use descriptive labels for the identified objects.
  """,
    output_key="object_detection_output",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
)

