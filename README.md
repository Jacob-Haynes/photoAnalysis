# Photo Analysis AI

This project is a multi-agent AI system designed to analyze images and determine the geographical location where they were taken. It uses an orchestrator agent to manage a workflow that includes image analysis and location identification by specialized sub-agents.

## Project Structure

- `core_agent/agent.py`: The main orchestrator agent that directs the analysis process.
- `sub_agents/`: Contains the specialized agents for specific tasks:
    - `image_analyst_agent.py`: Analyzes the content of the image.
    - `location_analyst_agent.py`: Identifies the geographical location from the image.
    - `object_detection_agent.py`: Detects objects within the image.
- `tools/`: Includes tools that the agents can use to perform their tasks.
- `requirements.txt`: A list of the Python dependencies required for this project.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables:**
    Create a `.env` file in the root directory to store your environment variables, such as API keys.

