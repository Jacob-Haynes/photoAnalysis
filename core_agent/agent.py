from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool

from sub_agents.image_analyst_agent import image_analyst_agent
from sub_agents.location_analyst_agent import location_analyst_agent

load_dotenv()

# --- Configuration ---
GCP_MODEL = "gemini-2.5-flash"

# --- Agent Definition ---

orchastrator_agent = LlmAgent(
    name="orchastrator",
    description="An AI agent that orchastrates an image analysis.",
    model=GCP_MODEL,
    instruction="""You are an expert at analysing an image.
Your goal is to analyse an image, provide a through writen analysis and then determine its geographical location.

1.  **Analyse the Image:** Use the `image_analyst_agent` to get a through analysis of the image which you will share to the user in full.
2.  **Deduce the Location:** Based on the identified elements, use the `location_analyst_agent` to deduce the country, city, and specific location where the image was taken. Ideally with a google maps link.
    """,
    tools=[agent_tool.AgentTool(agent=image_analyst_agent), agent_tool.AgentTool(agent=location_analyst_agent)],
    output_key="analysis_output",
)


# --- Agent Usage ---
root_agent = orchastrator_agent
