# Drone Security Analyst Agent Prototype (Object Detection + Q&A)

This repository contains the prototype for a Drone Security Analyst Agent, developed as part of the AI Engineer assignment. This version processes video frames using an Object Detection model (DETR) and simulated telemetry to monitor a property, identify objects, log events, generate alerts, index analysis results, and answer natural language questions about the findings.

## Feature Specification

**Value Proposition:**

This prototype provides property owners with enhanced situational awareness and security through automated, continuous monitoring. By analyzing drone video frames using AI Object Detection models and processing telemetry data, it delivers real-time alerts for potential threats and logs significant events, creating a searchable and queryable history for later review and analysis, reducing the need for constant manual observation.

**Key Functional Requirements:**

1.  **Video Frame & Telemetry Ingestion:** The system must ingest video frames (from a file or stream) and simulated time-stamped telemetry data (e.g., drone location).
2.  **Object Detection & Analysis:** The system must utilize an Object Detection model (e.g., DETR) to identify and locate objects (persons, vehicles, etc.) within video frames.
3.  **Event Logging & Alerting:** The system must log the detected objects with context (timestamp, location, object label, confidence score). It must generate immediate alerts based on predefined rules applied to the detected objects (e.g., person detected).
4.  **Persistent Detection Indexing & Querying:** The system must store object detection results (timestamp, location, list of detected objects with details) in a persistent database (SQLite) allowing querying based on various criteria (e.g., time range, object label, location).
5.  **(Bonus)** **Question Answering:** The system should allow users to ask natural language questions about the indexed detection history (e.g., "Were any trucks seen?").

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Pruthvi-Parade/aegis-drone-ai.git
    cd aegis-drone-ai
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    # Create the environment (use python3 if needed)
    python -m venv venv

    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Installs OpenCV, PyTorch, Transformers, Langchain, OpenAI client, python-dotenv, etc.)*

4.  **Sample Data:**
    *   **Option A (Use Provided Sample):** Download the sample video used during development from Mixkit: [Street View of a City Intersection](https://mixkit.co/free-stock-video/street-view-of-a-city-intersection-34601/).
    *   **Option B (Use Your Own Video):** You may use your own video file (e.g., MP4, AVI).
    *   **Required Action:** Create a `data/` directory in the project root. Place your chosen video file inside this directory and **rename it to `sample_video.mp4`**. The application currently expects the video at `data/sample_video.mp4`.

5.  **Setup OpenAI API Key:**
    *   Create a file named `.env` in the project root.
    *   Add your OpenAI API key:
        ```dotenv
        # .env
        OPENAI_API_KEY="your_openai_api_key_here"
        ```
    *(The Q&A feature requires a valid OpenAI API key)*

## Running the Prototype

To run the main agent simulation and Q&A:

```bash
python src/agent.py
