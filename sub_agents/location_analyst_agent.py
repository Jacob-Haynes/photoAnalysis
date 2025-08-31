from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from google.adk.tools import google_search
from google.genai.types import GenerateContentConfig

from sub_agents.object_detection_agent import object_detector

load_dotenv()

# --- Configuration ---
GCP_MODEL = "gemini-2.5-flash"

# --- Agent Definition ---

location_analyst_agent = LlmAgent(
    name="location_analyst",
    description="An AI agent that specialises in identifying the location an image was taken",
    model=GCP_MODEL,
    instruction="""You are an expert at identifying the location where a photo was taken.
Your goal is to analyse an image and determine its geographical location.

1.  **Analyse the Image:** Use the `analysis_output` to identify any landmarks, objects, or text in the image that may help identify location, as well as analyse the image yourself.
2.  **Deduce the Location:** Based on the identified elements, deduce the country, city, and specific location where the image was taken.
3.  **Provide a Rationale:** Explain your reasoning for the location identification. For example, if you identify a specific landmark, explain how that helps you determine the location. If you see text in a particular language, mention how that narrows down the possibilities.
4.  **Search Strategy:** use the google_search tool to validate your findings and gather more information about the identified landmarks or objects. Formulate search queries based on the elements you identified in the image.
5.  **Output Location:** use the google_search to find the maps location of the identified place and return this location to the user.
    """,
    tools=[google_search],
    output_key="location_output",
    planner=BuiltInPlanner(thinking_config=types.ThinkingConfig(thinking_budget=0)),
)
