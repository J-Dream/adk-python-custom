# Import necessary standard Python libraries
import os  # For interacting with the operating system (e.g., file paths)
import time  # For time-related functions (e.g., sleep, timestamps)
import json  # For working with JSON data (saving and loading logs)
import datetime  # For getting current date and time
import threading  # For running tasks in separate threads (non-blocking operations)

# Import libraries for Google Cloud Vertex AI
import vertexai  # Main Vertex AI SDK
# import vertexai.generative_models as genai_models  # For using generative AI models and creating content parts - Not directly used, Part is enough
from vertexai.generative_models import GenerativeModel, Part

# Import custom utility for video monitoring
from .video_utils import VideoMonitor  # Assumes video_utils.py is in the same 'app' directory

# Import library for loading environment variables from a .env file
from dotenv import load_dotenv

# --- Environment Variable Loading ---
# Construct the absolute path to the .env file, assuming it's in the 'streaming-test2/' directory (parent of 'app/')
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

class CameraAgent:
    """
    The CameraAgent is responsible for monitoring a video feed from a camera,
    detecting significant visual changes, sending these changes (as images)
    to a generative AI model for interpretation, logging the outcomes, and
    saving the latest captured frame for UI display.
    It runs its monitoring process in a separate background thread.
    """

    def __init__(self,
                 project_id: str,
                 location: str,
                 model_name: str = "gemini-2.0-flash",
                 camera_index: int = 0,
                 fps_limit: int = 1):
        """
        Initializes the CameraAgent.
        """
        if not project_id:
            raise ValueError("CAMERA_AGENT: Google Cloud Project ID is required for CameraAgent.")
        if not location:
            raise ValueError("CAMERA_AGENT: Google Cloud Location is required for CameraAgent.")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        self.log_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'logs',
            'camera_log.json'
        )
        # Define path for saving the latest captured image for the UI
        self.latest_image_path = os.path.join(
            os.path.dirname(__file__), # current directory (app)
            'static', # static sub-directory
            'latest_camera_image.jpg' # filename
        )

        # Ensure logs and static directories exist
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.latest_image_path), exist_ok=True) # Create static dir if not exists

        with open(self.log_file, 'w') as f: # Initialize/clear log file
            json.dump([], f)

        print(f"CAMERA_AGENT: Initializing Vertex AI with project: {self.project_id}, location: {self.location}")
        vertexai.init(project=self.project_id, location=self.location)

        print(f"CAMERA_AGENT: Loading generative model: {self.model_name}")
        self.model = GenerativeModel(self.model_name)
        self.chat = None

        self.video_monitor = VideoMonitor(camera_index=camera_index, fps_limit=fps_limit)
        self.running = False
        self.thread = None

        print(f"CAMERA_AGENT: Initialized. Logging to: {self.log_file}. Latest image will be at: {self.latest_image_path}")
        print(f"CAMERA_AGENT: Video monitor configured for camera index {camera_index}, FPS limit {fps_limit}.")

    def _start_chat_session(self):
        if not self.chat:
            print("CAMERA_AGENT: Starting new chat session with Vertex AI model...")
            try:
                self.chat = self.model.start_chat(response_validation=False)
                print("CAMERA_AGENT: Chat session started successfully.")
            except Exception as e:
                print(f"CAMERA_AGENT: Critical error starting chat session: {e}")
                self.chat = None
                raise

    def _log_change(self, description: str, model_comment: str, processing_time_ms: float = -1.0):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": "video_change",
            "description_by_cv": description,
            "comment_by_llm": model_comment,
            "llm_processing_time_ms": round(processing_time_ms, 2) # Log processing time, rounded
        }

        try:
            current_logs = []
            if os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0:
                with open(self.log_file, 'r') as f:
                    try:
                        current_logs = json.load(f)
                        if not isinstance(current_logs, list): current_logs = []
                    except json.JSONDecodeError: current_logs = []
            current_logs.append(log_entry)
            with open(self.log_file, 'w') as f:
                json.dump(current_logs, f, indent=4)
            # Include processing time in the console log for immediate feedback
            print(f"CAMERA_AGENT: Logged change. LLM comment: '{model_comment[:50]}...'. LLM Time: {processing_time_ms:.2f}ms")
        except IOError as e:
            print(f"CAMERA_AGENT: Error writing to log file {self.log_file}: {e}")
        except Exception as e:
            print(f'CAMERA_AGENT: Unexpected error during logging: {e}')

    def _monitor_loop(self):
        print("CAMERA_AGENT: Video monitoring loop started.")
        if not self.video_monitor.start_capture():
            print("CAMERA_AGENT: CRITICAL - Failed to start video capture. Monitoring loop cannot run.")
            self.running = False
            return

        while self.running:
            try:
                # frame_capture_time = time.time() # Not used for overall lag, LLM time is more specific
                changed, description, frame_bytes = self.video_monitor.process_frame_for_changes()

                # Save the latest frame if available for the UI, regardless of change detection for LLM
                if frame_bytes:
                    try:
                        with open(self.latest_image_path, 'wb') as f_img:
                            f_img.write(frame_bytes)
                        # print(f"CAMERA_AGENT: Latest image saved to {self.latest_image_path}") # Can be too noisy
                    except IOError as e:
                        print(f"CAMERA_AGENT: Error saving latest image {self.latest_image_path}: {e}")

                if changed and frame_bytes:
                    print(f"CAMERA_AGENT: Video change detected by CV: {description}. Preparing image for model.")

                    if not self.chat:
                        try:
                            self._start_chat_session()
                        except Exception as e:
                            print(f"CAMERA_AGENT: Could not start chat session for processing change: {e}. Skipping this event.")
                            time.sleep(5)
                            continue

                    image_part = Part.from_data(frame_bytes, mime_type="image/jpeg")
                    prompt_text = (
                        "You are an AI assistant observing a live video feed from a security camera. "
                        "A computer vision algorithm has detected a significant visual change in the scene. "
                        f"The algorithm's initial description of this change is: '{description}'. "
                        "Please analyze the provided image carefully and give a concise, human-readable description "
                        "of what you observe in the image that is relevant to this detected change. "
                        "Focus on new objects, significant movements, changes in object states, or anything unusual. "
                        "Avoid generic statements like 'I see an image.' Be specific."
                    )
                    print(f"CAMERA_AGENT: Sending video change event and image to model. Prompt: '{prompt_text[:100]}...'")

                    llm_start_time = time.time() # Time before sending to LLM
                    try:
                        response = self.chat.send_message([prompt_text, image_part])
                        llm_end_time = time.time() # Time after receiving from LLM
                        llm_processing_time_ms = (llm_end_time - llm_start_time) * 1000

                        model_comment = ""
                        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                            model_comment = response.candidates[0].content.parts[0].text or ""

                        if model_comment.strip():
                            self._log_change(description, model_comment, llm_processing_time_ms)
                        else:
                            print("CAMERA_AGENT: Model did not provide a text comment for the video change.")
                            self._log_change(description, "LLM provided no comment or an empty comment.", llm_processing_time_ms)

                    except Exception as e:
                        llm_end_time = time.time()
                        llm_processing_time_ms = (llm_end_time - llm_start_time) * 1000
                        print(f"CAMERA_AGENT: Error sending message to model or processing its response: {e} (LLM time: {llm_processing_time_ms:.2f}ms)")
                        self.chat = None
                        print("CAMERA_AGENT: Chat session reset due to error.")
                        time.sleep(2)

                elif changed:
                    print(f"CAMERA_AGENT: Video change detected ('{description}'), but frame data is unavailable. Cannot send to model.")

                time.sleep(0.1)

            except Exception as e:
                print(f'CAMERA_AGENT: Unexpected error in monitor loop: {e}')
                time.sleep(5)

        if self.video_monitor.cap:
            self.video_monitor.stop_capture()
        print("CAMERA_AGENT: Video monitoring loop stopped.")

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
            print("CAMERA_AGENT: Attempting to stop agent...")
            self.running = False

            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=10)
                if self.thread.is_alive():
                    print("CAMERA_AGENT: Warning - background thread did not stop in the allocated time (10 seconds).")
                else:
                    print("CAMERA_AGENT: Background thread stopped successfully.")
            else:
                print("CAMERA_AGENT: Background thread was not active or already stopped.")

            self.thread = None
            print("CAMERA_AGENT: Agent stopped.")
        else:
            print("CAMERA_AGENT: Agent is not currently running.")

if __name__ == '__main__':
    print("CAMERA_AGENT: Running camera_agent.py for direct testing...")
    project_id_env = os.getenv('GOOGLE_CLOUD_PROJECT')
    location_env = os.getenv('GOOGLE_CLOUD_LOCATION')
    use_vertex_env = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'

    if use_vertex_env and project_id_env and location_env:
        print(f"CAMERA_AGENT MAIN: Initializing agent with project='{project_id_env}', location='{location_env}', model='gemini-2.0-flash'")
        test_agent = CameraAgent(
            project_id=project_id_env,
            location=location_env,
            model_name="gemini-2.0-flash",
            camera_index=0,
            fps_limit=0.2
        )

        print("CAMERA_AGENT MAIN: Starting agent for testing...")
        test_agent.start()

        try:
            print("CAMERA_AGENT MAIN: Agent running. Monitoring for video changes for approximately 20 seconds.")
            print(f"CAMERA_AGENT MAIN: Check the log file '{test_agent.log_file}' for output.")
            print(f"CAMERA_AGENT MAIN: Check for latest image at '{test_agent.latest_image_path}'.") # Added this line
            time.sleep(20)
        except KeyboardInterrupt:
            print("CAMERA_AGENT MAIN: Test interrupted by user (KeyboardInterrupt).")
        finally:
            print("CAMERA_AGENT MAIN: Stopping agent...")
            test_agent.stop()
            print("CAMERA_AGENT MAIN: Direct test finished.")
    else:
        print("CAMERA_AGENT MAIN: CRITICAL - Environment variables for Vertex AI "
              "(GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) or GOOGLE_GENAI_USE_VERTEXAI=True "
              "not set correctly in .env file.")
        print(f"  Loaded Project ID: {project_id_env}")
        print(f"  Loaded Location: {location_env}")
        print(f"  Use Vertex AI: {use_vertex_env}")
        print(f"  The .env file was expected at: {dotenv_path}")
        print("CAMERA_AGENT MAIN: Skipping direct agent test.")
