import os
import time
import json
import datetime
import threading # For running the agent in the background
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from .video_utils import VideoMonitor # Assuming video_utils.py is in the same directory
from dotenv import load_dotenv

# Load environment variables from .env file in the root of streaming-test2
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=dotenv_path)

class CameraAgent:
    def __init__(self, project_id: str, location: str, model_name: str = "gemini-2.0-flash", camera_index: int = 0, fps_limit: int = 1):
        if not project_id:
            raise ValueError("Google Cloud Project ID is required.")
        if not location:
            raise ValueError("Google Cloud Location is required.")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'camera_log.json') # Correct path to logs/camera_log.json

        # Ensure logs directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        # Clear the log file at the start, or ensure it's a valid JSON array if appending
        with open(self.log_file, 'w') as f:
            json.dump([], f) # Initialize with an empty list for appending JSON objects

        print(f"CAMERA_AGENT: Initializing Vertex AI with project: {self.project_id}, location: {self.location}")
        vertexai.init(project=self.project_id, location=self.location)
        print(f"CAMERA_AGENT: Loading model: {self.model_name}")
        self.model = GenerativeModel(self.model_name)
        self.chat = None # Will be initialized when needed

        self.video_monitor = VideoMonitor(camera_index=camera_index, fps_limit=fps_limit)
        self.running = False
        self.thread = None

        print(f"CAMERA_AGENT: Initialized. Logging to: {self.log_file}")
        print(f"CAMERA_AGENT: Video monitor configured for camera index {camera_index}, FPS limit {fps_limit}.")

    def _start_chat_session(self):
        # Internal method to start or restart a chat session if needed
        if not self.chat:
            print("CAMERA_AGENT: Starting new chat session with Vertex AI...")
            try:
                self.chat = self.model.start_chat(response_validation=False)
                print("CAMERA_AGENT: Chat session started.")
            except Exception as e:
                print(f"CAMERA_AGENT: Error starting chat session: {e}")
                self.chat = None # Ensure chat is None if it fails
                raise # Re-raise the exception to signal failure

    def _log_change(self, description: str, model_comment: str):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": "video_change",
            "description_by_cv": description,
            "comment_by_llm": model_comment
        }

        # Read existing logs, append, and write back
        # This is not ideal for high frequency, but okay for this use case
        try:
            with open(self.log_file, 'r+') as f:
                try:
                    logs = json.load(f)
                except json.JSONDecodeError:
                    logs = [] # Initialize if file is empty or corrupt
                logs.append(log_entry)
                f.seek(0) # Rewind to the beginning of the file
                json.dump(logs, f, indent=4)
                f.truncate() # Remove remaining part of old file if new data is shorter
            print(f"CAMERA_AGENT: Logged change: {model_comment[:50]}...")
        except IOError as e:
            print(f"CAMERA_AGENT: Error writing to log file {self.log_file}: {e}")
        except json.JSONDecodeError as e:
             print(f"CAMERA_AGENT: Error decoding JSON from {self.log_file}: {e}")


    def _monitor_loop(self):
        print("CAMERA_AGENT: Monitor loop started.")
        if not self.video_monitor.start_capture():
            print("CAMERA_AGENT: Failed to start video capture. Exiting monitor loop.")
            self.running = False # Ensure we don't think we are running
            return

        while self.running:
            try:
                changed, description, frame_bytes = self.video_monitor.process_frame_for_changes()

                if changed and frame_bytes:
                    print(f"CAMERA_AGENT: Video change detected by CV: {description}. Preparing to send to model.")

                    if not self.chat:
                        try:
                            self._start_chat_session()
                        except Exception as e:
                            print(f"CAMERA_AGENT: Could not start chat session for processing change: {e}. Skipping this change.")
                            time.sleep(5) # Wait before retrying to avoid spamming errors
                            continue # Skip this iteration

                    image_part = Part.from_data(frame_bytes, mime_type="image/jpeg")
                    # More descriptive prompt
                    prompt_text = (
                        "You are an assistant observing a live video feed. "
                        "A significant visual change has just been detected by a computer vision algorithm. "
                        f"The CV algorithm described the change as: '{description}'. "
                        "Please analyze the provided image and give a concise, human-readable description of what you observe that might be relevant to this detected change. "
                        "Focus on new objects, significant movements, or changes in state. Avoid generic statements."
                    )

                    print(f"CAMERA_AGENT: Sending video change event to model with prompt: '{prompt_text[:100]}...'")

                    try:
                        # Send message without stream for this agent
                        response = self.chat.send_message([prompt_text, image_part])
                        model_comment = ""
                        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                            model_comment = response.candidates[0].content.parts[0].text or ""

                        if model_comment:
                            print(f"CAMERA_AGENT: Model commented on video change: '{model_comment}'")
                            self._log_change(description, model_comment)
                        else:
                            print("CAMERA_AGENT: Model did not provide a comment for the video change.")
                            # Optionally log that no comment was provided
                            # self._log_change(description, "No comment from LLM.")

                    except Exception as e:
                        print(f"CAMERA_AGENT: Error sending message to model or processing response: {e}")
                        # Consider if chat session needs reset
                        self.chat = None # Reset chat session on error
                        time.sleep(2) # Brief pause before next attempt

                elif changed: # Change detected but no frame_bytes
                    print(f"CAMERA_AGENT: Video change detected ({description}), but frame data is unavailable. Cannot send to model.")
                    # Optionally log this scenario
                    # self._log_change(description, "CV detected change, but frame data was not available for LLM.")

                # Respect FPS limit implicitly by video_monitor's own timing,
                # but add a small sleep to prevent tight loop if process_frame_for_changes is very fast
                # or if video_monitor.fps_limit is None.
                # The video_monitor.process_frame_for_changes() already has a sleep if fps_limit is set.
                # If it's not set, or if processing is faster than the limit, this adds a small yield.
                time.sleep(0.1) # Adjust as needed, ensures loop isn't too tight if no FPS limit in video_monitor

            except Exception as e:
                print(f"CAMERA_AGENT: Unexpected error in monitor loop: {e}")
                # Potentially add a backoff mechanism or stop the loop if errors persist
                time.sleep(5) # Wait a bit before trying again

        if self.video_monitor.cap:
            self.video_monitor.stop_capture()
        print("CAMERA_AGENT: Monitor loop stopped.")

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print("CAMERA_AGENT: Agent started monitoring in a background thread.")
        else:
            print("CAMERA_AGENT: Agent is already running.")

    def stop(self):
        if self.running:
            print("CAMERA_AGENT: Stopping agent...")
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10) # Wait for the thread to finish
            if self.thread and self.thread.is_alive():
                print("CAMERA_AGENT: Thread did not stop in time.")
            else:
                print("CAMERA_AGENT: Agent stopped.")
            self.thread = None
        else:
            print("CAMERA_AGENT: Agent is not running.")

# Main block for basic testing (optional, can be removed if run from a main orchestrator)
if __name__ == '__main__':
    print("CAMERA_AGENT: Running camera_agent.py for direct testing...")

    # These should be loaded from .env in the root of streaming-test2
    # The load_dotenv at the top of the file should handle this.
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION')
    use_vertex = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'

    if use_vertex and project and location:
        print(f"CAMERA_AGENT MAIN: Initializing agent with project='{project}', location='{location}', model='gemini-2.0-flash'")
        agent = CameraAgent(
            project_id=project,
            location=location,
            model_name="gemini-2.0-flash", # Specify model, could also come from config.py
            camera_index=0,
            fps_limit=0.2 # Process one frame every 5 seconds for testing
        )

        print("CAMERA_AGENT MAIN: Starting agent...")
        agent.start()

        try:
            print("CAMERA_AGENT MAIN: Agent running. Monitoring for video changes for ~20 seconds.")
            print("CAMERA_AGENT MAIN: Check 'streaming-test2/logs/camera_log.json' for output.")
            time.sleep(20)
        except KeyboardInterrupt:
            print("CAMERA_AGENT MAIN: Test interrupted by user.")
        finally:
            print("CAMERA_AGENT MAIN: Stopping agent...")
            agent.stop()
            print("CAMERA_AGENT MAIN: Direct test finished.")
    else:
        print("CAMERA_AGENT MAIN: Environment variables for Vertex AI (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) or GOOGLE_GENAI_USE_VERTEXAI not set correctly in .env. Skipping agent direct test.")
        print(f"Project: {project}, Location: {location}, Use Vertex: {use_vertex}")
        print(f"Make sure .env is in the streaming-test2/ directory and contains the correct values.")
