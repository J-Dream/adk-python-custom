import os
import time # Required for sleep in __main__
import vertexai
from vertexai.generative_models import GenerativeModel, Part

from .audio_utils import transcribe_audio_bytes, synthesize_text_to_audio_bytes, play_audio_bytes, record_audio
from .video_utils import VideoMonitor
from dotenv import load_dotenv # Added for direct run __main__

class MultimodalAgent:
    def __init__(self, project_id: str, location: str, model_name: str, camera_index: int = 0):
        if not project_id:
            raise ValueError("Google Cloud Project ID is required.")
        if not location:
            raise ValueError("Google Cloud Location is required.")
        if not model_name:
            raise ValueError("Model name is required.")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name

        vertexai.init(project=self.project_id, location=self.location)
        self.model = GenerativeModel(self.model_name)
        self.chat = None

        self.video_monitor = VideoMonitor(camera_index=camera_index, fps_limit=1) # Process 1 frame per second for changes

        print(f"MultimodalAgent initialized with project: {self.project_id}, location: {self.location}, model: {self.model_name}")
        print(f"Video monitor configured for camera index {camera_index}.")

    def start_chat(self): # Renamed from start_agent to align with Vertex AI SDK's chat concept
        self.chat = self.model.start_chat(response_validation=False)
        print("New chat session started with Vertex AI.")
        if not self.video_monitor.cap:
             if self.video_monitor.start_capture():
                 print("Video capture started successfully.")
             else:
                 print("Failed to start video capture.")
        return self.chat

    def stop_chat(self):
        if self.video_monitor.cap:
            self.video_monitor.stop_capture()
        self.chat = None
        print("Chat session ended and video capture stopped.")

    def send_text_message(self, text_prompt: str) -> str:
        if not self.chat:
            self.start_chat()

        print(f"Sending text to model: {text_prompt}")
        try:
            response = self.chat.send_message(text_prompt)
            return response.text
        except Exception as e:
            print(f"Error sending text message to model: {e}")
            raise

    def handle_voice_interaction(self, record_duration_seconds: int = 5, event_queue=None):
        if not self.chat:
            self.start_chat()

        print("\n--- Starting Voice Interaction Cycle ---")
        user_audio_bytes = record_audio(duration_seconds=record_duration_seconds)
        if not user_audio_bytes:
            print("Audio recording failed. Aborting voice interaction.")
            if event_queue: event_queue.put({"type": "status", "message": "Audio recording failed."})
            return

        user_text_prompt = transcribe_audio_bytes(user_audio_bytes)
        if user_text_prompt is None:
            print("Audio transcription failed. Aborting voice interaction.")
            if event_queue: event_queue.put({"type": "status", "message": "Transcription failed."})
            return
        if not user_text_prompt:
            print("No speech detected in the recording.")
            if event_queue: event_queue.put({"type": "status", "message": "No speech detected."})
            # Could send "I didn't catch that" audio response here
            return

        print(f"User (transcribed): {user_text_prompt}")
        if event_queue: event_queue.put({"type": "transcription", "text": user_text_prompt})

        model_response_text = self.send_text_message(user_text_prompt)
        if not model_response_text:
            print("Model did not return a text response. Aborting voice interaction.")
            if event_queue: event_queue.put({"type": "status", "message": "Model did not respond."})
            return

        print(f"Model (text): {model_response_text}")
        if event_queue: event_queue.put({"type": "model_response_audio_text", "text": model_response_text})

        model_audio_bytes = synthesize_text_to_audio_bytes(model_response_text)
        if not model_audio_bytes:
            print("Model audio synthesis failed. Cannot play back response.")
            return

        play_audio_bytes(model_audio_bytes, output_format="mp3") # Plays locally on server
        print("--- Voice Interaction Cycle Ended ---")


    def check_for_video_changes_and_comment(self, event_queue=None):
        if not self.video_monitor.cap or not self.video_monitor.cap.isOpened():
            # Try to start it if it's not running but should be (e.g. chat is active)
            if self.chat: # if a chat is supposed to be active, video should be too
                print("Video monitor not active, attempting to restart for change detection...")
                self.video_monitor.start_capture()
                if not self.video_monitor.cap or not self.video_monitor.cap.isOpened():
                    print("Video capture is not active. Cannot check for changes.")
                    return
            else: # No active chat, no need to check for changes to send to model
                return


        changed, description, frame_bytes = self.video_monitor.process_frame_for_changes()

        if changed and frame_bytes:
            print(f"Video change detected: {description}. Sending to model.")

            if not self.chat: # Ensure chat is started before sending a message to the model
                self.start_chat()

            image_part = Part.from_data(frame_bytes, mime_type="image/jpeg")
            prompt_text = f"A change was detected in the video feed: {description}. Concisely describe what you observe in this image related to the change."

            model_response = self.send_message(text_prompt=prompt_text, image_parts=[image_part])

            if model_response:
                print(f"Model commented on video change: {model_response}")
                if event_queue:
                    event_queue.put({"type": "video_change_comment", "comment": model_response})
            else:
                print("Model did not provide a comment for the video change.")
        elif changed:
            print(f"Video change detected ({description}), but frame data is unavailable.")

    def send_message(self, text_prompt: str = None, image_parts: list = None, audio_parts: list = None, video_parts: list = None) -> str:
        if not self.chat:
            self.start_chat()

        if not text_prompt and not image_parts and not audio_parts and not video_parts:
            raise ValueError("At least one input type (text, image, audio, video) must be provided.")

        prompt_parts = []
        if text_prompt:
            prompt_parts.append(Part.from_text(text_prompt))

        if image_parts: prompt_parts.extend(image_parts)
        if audio_parts: prompt_parts.extend(audio_parts)
        if video_parts: prompt_parts.extend(video_parts)

        # print(f"Sending parts to model. Text: '{text_prompt if text_prompt else ''}'. Images: {len(image_parts) if image_parts else 0}.")
        print(f"Sending parts to model: {[(type(p.to_dict().get('inline_data', {}).get('mime_type', 'text')) if not p.to_dict().get('text', None) else 'text') for p in prompt_parts]}")


        try:
            response = self.chat.send_message(prompt_parts, stream=False)
            return response.text
        except Exception as e:
            print(f"Error sending message to model: {e}")
            raise

if __name__ == '__main__':
    print("Attempting to run agent.py for video change detection test...")
    # This __main__ block is for direct testing of agent.py.
    # For web app, use run.py.
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')) # For direct run
    use_vertex_env = os.getenv('GOOGLE_GENAI_USE_VERTEXAI', 'False').lower() == 'true'
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION')
    model_name_config = "gemini-2.0-flash-live-preview-04-09"

    if use_vertex_env and project and location:
        print(f"Initializing agent with project='{project}', location='{location}', model='{model_name_config}'")
        agent = MultimodalAgent(
            project_id=project,
            location=location,
            model_name=model_name_config,
            camera_index=0
        )
        agent.start_chat()

        if not agent.video_monitor.cap or not agent.video_monitor.cap.isOpened():
            print("Agent started, but video capture failed. Video tests will be skipped.")
        else:
            print("Agent started with video capture. Monitoring for video changes for ~15 seconds.")
            start_time = time.time()
            try:
                while time.time() - start_time < 15:
                    agent.check_for_video_changes_and_comment() # No event queue in this direct test
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Test interrupted.")
            finally:
                agent.stop_chat()
    else:
        print("Environment variables for Vertex AI not set. Skipping agent video test.")
