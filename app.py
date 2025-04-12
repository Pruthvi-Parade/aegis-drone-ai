# app.py

import streamlit as st
import time
import datetime
import cv2
import pandas as pd # For displaying query results
from PIL import Image

# Import necessary components from our src directory
from src.simulator import generate_drone_data, VIDEO_PATH
from src.vlm_processor import VLMProcessor
from src.indexer import FrameIndexerSQLite
from src.agent import extract_info_from_description # Reuse our simple extractor

# --- Page Configuration ---
st.set_page_config(page_title="Drone Security Agent", layout="wide")
st.title("👁️ Drone Security Analyst Agent")
st.caption("A prototype using VLM (BLIP), SQLite Indexing, and Streamlit")

# --- Initialization & Caching ---
# Use Streamlit's caching to load models/DB connections only once.
# @st.cache_resource
# def initialize_vlm():
#     st.write("Cache miss: Initializing VLM Processor...")
#     return VLMProcessor()

@st.cache_resource
def initialize_indexer():
     st.write("Cache miss: Initializing Frame Indexer (SQLite)...")
     return FrameIndexerSQLite()

# vlm_processor = initialize_vlm()
indexer = initialize_indexer()

# Initialize session state variables if they don't exist
if 'agent_running' not in st.session_state:
    st.session_state.agent_running = False
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = ["App Initialized."]
if 'alert_messages' not in st.session_state:
    st.session_state.alert_messages = []
if 'frame_count' not in st.session_state:
     st.session_state.frame_count = 0
if 'simulator' not in st.session_state:
     # Create the generator, store it in session state to persist across reruns
     # We'll recreate it when starting processing
     st.session_state.simulator = None


# --- UI Layout ---
sidebar = st.sidebar
control_col, display_col = st.columns([1, 2]) # Main display column, Control column

# --- Sidebar Controls ---
sidebar.header("Controls")
max_frames_to_process = sidebar.number_input("Max Frames to Process per Run", min_value=1, max_value=100, value=10)
frame_skip_config = sidebar.number_input("Process Every Nth Frame (Simulator)", min_value=1, max_value=100, value=30) # Higher skip for faster UI demo

if sidebar.button("Start Processing", key="start_btn", disabled=st.session_state.agent_running):
    st.session_state.agent_running = True
    st.session_state.log_messages.append(f"START: Processing up to {max_frames_to_process} frames...")
    st.session_state.frame_count = 0
    # Ensure VLM and Indexer are ready (they are cached)
    if not vlm_processor.model:
        st.error("VLM model not loaded. Cannot start.")
        st.session_state.agent_running = False
    else:
        # Create a new simulator generator to restart the video
        st.session_state.simulator = generate_drone_data(video_path=VIDEO_PATH, frame_skip=frame_skip_config)
        st.rerun() # Start the processing loop on the next run

if sidebar.button("Stop Processing", key="stop_btn", disabled=not st.session_state.agent_running):
    st.session_state.agent_running = False
    st.session_state.log_messages.append("STOP: Processing stopped by user.")
    st.session_state.simulator = None # Clear generator
    st.rerun() # Update UI state

sidebar.divider()
sidebar.header("Status")
sidebar.metric("Agent Status", "Running" if st.session_state.agent_running else "Stopped")
sidebar.metric("Frames Processed (this run)", st.session_state.frame_count)
sidebar.metric("Total Indexed Frames", indexer.get_index_size())


# --- Main Display Area ---
with display_col:
    st.header("Live Analysis")
    frame_placeholder = st.empty()
    analysis_placeholder = st.container() # Use container for description + keywords

    st.divider()
    st.header("Logs & Alerts")
    log_col, alert_col = st.columns(2)
    with log_col:
        st.subheader("Event Log")
        log_area = st.text_area("Logs", value="\n".join(st.session_state.log_messages[-20:]), height=200, key="log_disp", disabled=True)
    with alert_col:
         st.subheader("Alerts")
         alert_area = st.text_area("Alerts", value="\n".join(st.session_state.alert_messages[-15:]), height=200, key="alert_disp", disabled=True)

# --- Processing Loop (runs when agent_running is True) ---
if st.session_state.agent_running:
    if st.session_state.simulator is None:
        st.error("Simulator not ready. Please Start Processing.")
        st.session_state.agent_running = False # Safety stop
    else:
        try:
            # Check frame limit
            if st.session_state.frame_count >= max_frames_to_process:
                st.session_state.log_messages.append(f"FINISH: Reached processing limit ({max_frames_to_process} frames).")
                st.session_state.agent_running = False
                st.session_state.simulator = None # Clear generator
                st.rerun()

            # Get next frame
            timestamp, location, frame_bgr = next(st.session_state.simulator)
            st.session_state.frame_count += 1

            # --- Perform Analysis (VLM + Extraction) ---
            analysis_start_time = time.time()
            vlm_description = "Error: VLM Not Initialized" # Default value
            extracted_info = {} # Default value
            detected_objects = [] # Default value
            potential_actions = [] # Default value

            if vlm_processor: # <<< ADD THIS CHECK
                 # Only run analysis if vlm_processor was successfully initialized
                 try:
                     vlm_description = vlm_processor.analyze_frame(frame_bgr)
                     extracted_info = extract_info_from_description(vlm_description)
                     detected_objects = extracted_info.get('detected_objects', [])
                     potential_actions = extracted_info.get('potential_actions', [])
                 except Exception as analysis_error:
                     vlm_description = f"VLM Analysis Error: {analysis_error}"
                     st.error(f"Error during VLM analysis: {analysis_error}") # Show error in UI
            else: # <<< ADD THIS ELSE
                 # If VLM is not initialized (our current test case)
                 vlm_description = "VLM Disabled (Testing)"

            analysis_duration = time.time() - analysis_start_time # Calculate duration anyway
            
            # --- Display Current Frame ---
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            caption = f"Frame {st.session_state.frame_count} @ {timestamp.strftime('%H:%M:%S')} | Analysis: {analysis_duration:.2f}s"
            frame_placeholder.image(frame_rgb, caption=caption, use_column_width=True)

            # --- Display Analysis Results ---
            with analysis_placeholder: # Update content within the container
                st.markdown(f"**VLM Description:**")
                st.info(vlm_description) # Use info box for description
                st.write({
                    "Detected Objects": detected_objects or "None",
                    "Potential Actions": potential_actions or "None"
                })

            # --- Log Event ---
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            obj_str = ", ".join(detected_objects) or "None"
            act_str = ", ".join(potential_actions) or "None"
            log_msg = f"LOG [{ts_str}] @ {location} | VLM: '{vlm_description}' | Obj: [{obj_str}] | Act: [{act_str}]"
            st.session_state.log_messages.append(log_msg)

            # --- Check Alerts ---
            is_night = timestamp.hour >= 22 or timestamp.hour < 6
            alert_triggered = False
            if 'person' in detected_objects or 'pedestrian' in detected_objects:
                 alert_msg = f"ALERT [{ts_str}] Person/Pedestrian detected at {location}."
                 st.session_state.alert_messages.append(alert_msg)
                 alert_triggered = True
            # Add other rules here...

            if alert_triggered: # Visually indicate alert in UI
                 analysis_placeholder.warning("ALERT TRIGGERED (See Alerts Panel)")


            # --- Index Data ---
            try:
                indexer.add_frame_analysis(
                    timestamp=timestamp, location=location, vlm_description=vlm_description,
                    frame_number=st.session_state.frame_count,
                    detected_objects=detected_objects, potential_actions=potential_actions
                )
            except Exception as e:
                 log_msg = f"ERROR [{ts_str}] Failed to index frame: {e}"
                 st.session_state.log_messages.append(log_msg)
                 st.error(f"Indexing Error: {e}") # Show error in main UI too

            # --- Trigger UI Update for next iteration ---
            # Short sleep helps ensure UI updates smoothly between iterations
            time.sleep(0.05)
            st.rerun() # Rerun the script to process the next frame

        except StopIteration:
            st.session_state.log_messages.append("FINISH: Simulator reached end of video.")
            st.session_state.agent_running = False
            st.session_state.simulator = None # Clear generator
            st.warning("Video processing finished.")
            st.rerun() # Update UI state
        except Exception as e:
            st.session_state.log_messages.append(f"ERROR: {e}")
            st.session_state.agent_running = False
            st.session_state.simulator = None # Clear generator
            st.error(f"An error occurred during processing: {e}")
            # Consider adding full traceback logging for debugging
            # import traceback
            # st.session_state.log_messages.append(traceback.format_exc())
            st.rerun() # Update UI state

# --- Query Interface ---
st.divider()
st.header("Query Indexed Frames")

q_col1, q_col2, q_col3, q_col4 = st.columns(4)
with q_col1:
    q_object = st.text_input("Object Detected", key="q_obj")
    q_action = st.text_input("Action Detected", key="q_act") # Added action query
with q_col2:
    q_location = st.text_input("Location", key="q_loc")
    q_description = st.text_input("Description Contains", key="q_desc")
with q_col3:
     q_start_date = st.date_input("Start Date", value=None, key="q_start_d")
     q_start_time = st.time_input("Start Time", value=None, key="q_start_t")
with q_col4:
     q_end_date = st.date_input("End Date", value=None, key="q_end_d")
     q_end_time = st.time_input("End Time", value=None, key="q_end_t")

q_limit = st.slider("Max Results", 10, 500, 50, key="q_limit")

if st.button("Search Index", key="search_btn"):
    query_params = {"limit": q_limit}
    if q_object: query_params["object_detected"] = q_object
    # Modify indexer query_frames if it needs to handle actions separately,
    # or use description_contains for now if actions are just keywords.
    # Assuming basic keyword check for actions in description for now:
    if q_action: query_params["description_contains"] = q_action
    if q_location: query_params["location"] = q_location
    if q_description: query_params["description_contains"] = q_description # Overwrites action if both set - refine query logic if needed

    start_datetime, end_datetime = None, None
    if q_start_date and q_start_time: start_datetime = datetime.datetime.combine(q_start_date, q_start_time)
    if q_end_date and q_end_time: end_datetime = datetime.datetime.combine(q_end_date, q_end_time)
    if start_datetime: query_params["start_time"] = start_datetime
    if end_datetime: query_params["end_time"] = end_datetime

    st.write(f"Running query with: `{query_params}`")
    with st.spinner("Querying database..."):
         results = indexer.query_frames(**query_params)

    st.write(f"Found **{len(results)}** results.")
    if results:
        df = pd.DataFrame(results)
        # Select and reorder columns for display
        display_cols = ['timestamp', 'location', 'vlm_description', 'detected_objects', 'potential_actions', 'id']
        df_display = df[[col for col in display_cols if col in df.columns]]
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No matching frames found.")