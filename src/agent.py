# src/agent.py

import time
import datetime
from typing import Dict, Any, List, Optional
import os # For loading API keys

# --- Import Components ---
from simulator import generate_drone_data, VIDEO_PATH
from vlm_processor import ObjectDetector
from indexer import FrameIndexerSQLiteOD

# --- LangChain ---
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# --- Q&A Specific Imports ---
from langchain_openai import ChatOpenAI # LLM
from langchain_openai import OpenAIEmbeddings # Embeddings (Optional for simple retrieval)
from langchain.schema import Document # To format retrieved data
from dotenv import load_dotenv # To load API key

# --- Load Environment Variables ---
load_dotenv()
# Check if the key is loaded (optional but good practice)
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set. Q&A feature will not work.")


class DroneSecurityAgentOD:
    def __init__(self):
        """Initializes the agent with OD, Indexer, and QA components."""
        print("Initializing Drone Security Agent (Object Detection + QA)...")
        self.object_detector = ObjectDetector()
        # Use the main OD database for the agent run
        self.indexer = FrameIndexerSQLiteOD(db_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'security_analysis_od.db'))
        self.simulator = generate_drone_data(video_path=VIDEO_PATH, frame_skip=20)

        # --- OD Processing Chain (Unchanged) ---
        self.processing_chain = (
            RunnablePassthrough.assign( # Keep original data, add detections key
                detections=RunnableLambda(lambda x: self.object_detector.detect_objects(x['frame']))
            ) |
            RunnableLambda(self._process_frame_result_od) # <<< ENSURE THIS NAME MATCHES THE METHOD DEFINITION
        )

        # --- Initialize QA Components ---
        # Only initialize if API key is available
        self.llm = None
        self.qa_retriever = None
        self.qa_chain = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                # Define a simple retriever function that uses our indexer
                self.qa_retriever = RunnableLambda(self._retrieve_context_for_qa)
                # Define the QA prompt
                template = """Answer the question based only on the following context from drone footage analysis:
                Context:
                {context}

                Question: {question}

                Answer:"""
                prompt = PromptTemplate.from_template(template)

                # Define the QA Chain
                self.qa_chain = (
                    RunnableParallel(
                        {"context": self.qa_retriever, "question": RunnablePassthrough()} # Pass question through
                    )
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                print("QA Chain initialized successfully.")
            except Exception as e:
                print(f"Error initializing QA components (LLM/Chain): {e}. QA Feature may be disabled.")
                self.qa_chain = None # Disable QA if setup fails
        else:
             print("QA Feature disabled due to missing OpenAI API Key.")


        print("LangChain processing sequence (Object Detection) built.")
        print("Drone Security Agent Initialized.")
        print("-" * 30)


    # --- OD Processing Methods (_log_event_od, _check_alerts_od, _process_frame_result_od) ---
    # ... (Keep these methods exactly as they were in the previous step) ...
    def _log_event_od(self, analysis_result: Dict[str, Any]):
        """Formats and logs significant events based on object detections."""
        timestamp_str = analysis_result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        location = analysis_result['location']
        detections = analysis_result.get('detections', [])
        if not detections: return
        summary = [f"{det.get('label', '?')}({det.get('score', 0):.2f})" for det in detections]
        log_message = f"LOG [{timestamp_str}] Event at {location}. Detections: [{', '.join(summary)}]."
        print(log_message)

    def _check_alerts_od(self, analysis_result: Dict[str, Any]):
        """Checks detection results against rules and generates alerts."""
        timestamp = analysis_result['timestamp']
        location = analysis_result['location']
        detections = analysis_result.get('detections', [])
        detected_labels = {det.get('label') for det in detections if 'label' in det}
        if 'person' in detected_labels:
            alert_message = f"ALERT [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Person detected at {location}!"
            print(alert_message)
        if 'truck' in detected_labels:
             alert_message = f"ALERT [{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Truck detected at {location}."
             print(alert_message)
        # Add more rules if needed...

    def _process_frame_result_od(self, analysis_result: Dict[str, Any]):
        """Final step: Logs, Alerts, and Indexes the object detection results."""
        # 1. Log the event
        self._log_event_od(analysis_result)

        # 2. Check for alerts
        self._check_alerts_od(analysis_result)

        # 3. Add to indexer
        try:
            # --- CORRECTED CALL ---
            self.indexer.add_detections(
                timestamp=analysis_result['timestamp'],
                location=analysis_result['location'], # Pass location
                detections=analysis_result.get('detections', []), # Pass detections list
                frame_number=analysis_result.get('frame_number')
            )
            # --- END CORRECTION ---
        except Exception as e:
            print(f"Agent Error: Failed to index frame detections: {e}")
            # Consider logging traceback for detailed debugging if errors persist
            # import traceback
            # traceback.print_exc()

        return analysis_result # Pass through if needed

    # --- Run Method (Unchanged) ---
    def run(self, max_frames: Optional[int] = 50):
        # ... (Keep this method exactly as it was in the previous step) ...
        print(f"Agent starting run (Object Detection, max_frames={max_frames})...")
        frame_counter = 0
        if not self.simulator: print("Error: Simulator not initialized."); return
        if not self.object_detector.model: print("Error: Object Detector Model not loaded."); return
        start_run_time = time.time()
        try:
            for timestamp, location, frame in self.simulator:
                # ... process frame using self.processing_chain ...
                if max_frames is not None and frame_counter >= max_frames: break
                frame_counter += 1
                print(f"\n--- Processing Frame {frame_counter} ({timestamp.strftime('%H:%M:%S')}) [OD] ---")
                start_frame_time = time.time()
                chain_input = {"timestamp": timestamp, "location": location, "frame": frame, "frame_number": frame_counter}
                try:
                    result = self.processing_chain.invoke(chain_input)
                    end_frame_time = time.time()
                    print(f"Frame {frame_counter} processed in {end_frame_time - start_frame_time:.2f} seconds.")
                except Exception as e: print(f"!! Error processing frame {frame_counter} with LangChain OD: {e}"); continue
        except KeyboardInterrupt: print("\nAgent run interrupted by user.")
        except Exception as e: print(f"\nAn unexpected error during agent run: {e}"); import traceback; traceback.print_exc()
        finally:
            end_run_time = time.time(); print("-" * 30)
            final_index_size = self.indexer.get_index_size()
            print(f"Agent run finished after processing {frame_counter} frames in {end_run_time - start_run_time:.2f} seconds.")
            print(f"Total frames indexed (OD): {final_index_size}"); print("-" * 30)

    # --- Query Method (Unchanged) ---
    def search_indexed_detections(self, **kwargs) -> List[Dict[str, Any]]:
        # ... (Keep this method exactly as it was in the previous step) ...
        print(f"\nSearching OD index with criteria: {kwargs}")
        results = self.indexer.query_detections(**kwargs)
        print(f"Found {len(results)} matching frame records.")
        for i, event in enumerate(results[:10]):
             dets_summary = ", ".join([f"{d.get('label','?')}({d.get('score',0):.2f})" for d in event.get('detected_objects',[])])
             print(f"  Result {i+1}: {event['timestamp']} at {event['location']} - Detections: [{dets_summary}]")
        if len(results) > 10: print("  ...")
        return results

    # --- NEW: Q&A Functionality ---
    def _retrieve_context_for_qa(self, question: str) -> str:
        question_lower = question.lower()
        keywords = [word for word in question_lower.split() if len(word) > 3 and word not in [...]]
        known_objects = [...]
        object_keywords = [word for word in keywords if word in known_objects]

        singular_map = {...}
        target_singular_labels = set()
        for kw in object_keywords: # This uses object_keywords, which depends on 'keywords'
             target_singular_labels.add(singular_map.get(kw, kw))
        print(f"  -> Identified object keywords: {object_keywords}. Matching singular labels: {target_singular_labels}")
        # --- END NEW MAPPING ---


        general_context_limit = 30
        retrieved_docs = self.indexer.query_detections(limit=general_context_limit)

        # Filter these results further based on keywords in Python
        filtered_docs = []
        if target_singular_labels: # Check if we have target labels
             for doc in retrieved_docs:
                 # Get lowercase labels detected in this frame
                 doc_labels = {det.get('label', '').lower() for det in doc.get('detected_objects', [])}
                 # --- MODIFIED CHECK ---
                 # Check if any target singular label intersects with this frame's detected labels
                 if not target_singular_labels.isdisjoint(doc_labels): # Check for intersection
                     filtered_docs.append(doc)
             # --- END MODIFIED CHECK ---
             print(f"  -> Found {len(filtered_docs)} frames containing labels: {target_singular_labels}")
        else:
             # If no specific objects mentioned, use the most recent general frames
             filtered_docs = retrieved_docs
             print("  -> No specific object keywords found, using general recent context.")


        if not filtered_docs:
            print("  -> No relevant frames found in index based on criteria.")
            return "No relevant information found in the drone footage index."

        # Format the filtered data as context string (Limit context size for LLM)
        max_context_frames = 10 # Limit how many frames we send to LLM
        context_str = "Frames Analysis (Most Recent Relevant First):\n"
        for doc in filtered_docs[:max_context_frames]: # Take the top N filtered results
            ts = doc.get('timestamp', datetime.datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            loc = doc.get('location', 'Unknown')
            dets = doc.get('detected_objects', [])
            dets_summary = ", ".join([f"{d.get('label','?')}(score={d.get('score',0):.2f})" for d in dets])
            if not dets_summary: dets_summary = "None" # Explicitly show if no detections
            context_str += f"- Time: {ts}, Location: {loc}, Detections: [{dets_summary}]\n"

        print(f"  -> Retrieved context (showing first 500 chars):\n{context_str[:500]}...")
        return context_str

    def answer_question(self, question: str) -> str:
        """Answers a natural language question using the QA chain."""
        if not self.qa_chain:
            return "Q&A system is not initialized (likely missing API key or setup error)."

        print(f"\nQA - Answering question: '{question}'")
        try:
            # Invoke the QA chain
            answer = self.qa_chain.invoke(question)
            print(f"QA - Generated Answer: {answer}")
            return answer
        except Exception as e:
             print(f"QA - Error during QA chain invocation: {e}")
             return f"Error answering question: {e}"


# --- Main Execution ---
if __name__ == "__main__":
    agent = DroneSecurityAgentOD()
    # Run the agent to populate the database (optional, can comment out if DB exists)
    print("\n--- Running Agent to Populate Index ---")
    # agent.run(max_frames=20) # Run for a few frames

    # --- Example Q&A ---
    print("\n--- Answering Questions ---")
    q1 = "Were any people detected?"
    ans1 = agent.answer_question(q1)
    print(f"\nQ: {q1}\nA: {ans1}")

    q2 = "How many cars were seen?" # Note: LLM might estimate based on context
    ans2 = agent.answer_question(q2)
    print(f"\nQ: {q2}\nA: {ans2}")

    q3 = "Tell me about detected suitcases"
    ans3 = agent.answer_question(q3)
    print(f"\nQ: {q3}\nA: {ans3}")

    q4 = "Was anything detected near the Main Gate?" # Test filtering (won't find if location fixed)
    ans4 = agent.answer_question(q4)
    print(f"\nQ: {q4}\nA: {ans4}")