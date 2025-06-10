import os # For interacting with the OS
import json # For JSON data handling
import base64 # For encoding/decoding binary data for JSON
import datetime # For timestamping log entries
import warnings # For handling warnings
import asyncio # For asynchronous programming, crucial for ADK streaming
from pathlib import Path # For object-oriented file path handling
from dotenv import load_dotenv # For loading environment variables from a “.env” file

# Imports from the Google Agent Development Kit (ADK)
from google.genai.types import Part, Content, Blob # Data structures for content in ADK
from google.adk.runners import InMemoryRunner # Runner for executing ADK agents in memory
from google.adk.agents import LiveRequestQueue, Agent # Core ADK Agent class and live request handling
from google.adk.agents.run_config import RunConfig # Configuration for how an agent runs

# --- Environment Variable Loading ---
# Set the path to the .env file, expecting it in the `streaming-test2` directory.
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Warnings filtering - suppress specific UserWarnings, often from pydantic used by FastAPI
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "ADK_Streaming_Audio_Agent" # A name for this application/module, used by ADK

class AudioAgent:
    """
    The AudioAgent handles live bi-directional audio streaming using the Google ADK.
    It integrates with a FastAPI backend to communicate with a client (e.g., a web browser).
    It manages ADK sessions, processes incoming and outgoing audio/text, logs
    transcriptions, and potentially incorporates context from other agents.
    """

    def __init__(self,project_id: str, location: str, model_name: str = "gemini-2.0-flash-live-preview-04-09"):
        """
        Initializes the AudioAgent.

        Args:
            project_id (str): The Google Cloud Project ID.
            location (str): The Google Cloud region for Vertex AI services.
            model_name (str, optional): The name of the Gemini model to use for live audio.
                                        Defaults to "gemini-2.0-flash-live-preview-04-09".
        """
        # Validate project ID and location, they are essential.
        if not project_id:
            raise ValueError("CAUDIO_AGENT: Google Cloud Project ID is required.")
        if not location:
            raise ValueError("CAUDIO_AGENT: Google Cloud Location is required.")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name # Store original model name for reference

        # Ensure that the application is configured to use Vertex AI.
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() != "true":
            raise EnvironmentError("AUDIO_AGENT: GOOGLE_GENAI_USE_VERTEXAI must be set to True in .env")

        # Define the base ADK Agent object. The instruction will be
        # dynamically updated with camera context in start_adk_session.
        # This is a template or default agent definition.
        self.base_agent_description = "Live audio conversational agent."
        self.base_agent_instruction = "You are a helpful voice assistant. Respond to the user's speech. Be concise and natural." # Camera context will be appended
        print(f"AUDIO_AGENT: Base ADK agent configured with model: {model_name}")

        # Define paths for log files.
        self.audio_log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'audio_log.json')
        self.camera_log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'camera_log.json')

        # Ensure the 'logs' directory exists.
        os.makedirs(os.path.dirname(self.audio_log_file), exist_ok=True)
        with open(self.audio_log_file, 'w') as f:
            json.dump([], f)
        print(f"AUDIO_AGENT: Logging audio transcriptions to: {self.audio_log_file}")

        if not os.path.exists(self.camera_log_file):
            with open(self.camera_log_file, 'w') as f:
                json.dump([],f)
            print(f"AUDIO_AGENT: Initialized empty camera log file at: {self.camera_log_file}")

        self.active_adk_sessions = {}
        self.camera_context = "No camera updates yet."

    def _log_transcription(self,speaker: str, text: str):
        """
        Internal helper method to log audio transcriptions to a JSON file.
        Each log entry includes a timestamp, the speaker ('user' or 'agent'), and the transcribed text.
        """
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "speaker": speaker, "text": text}
        try:
            current_logs = []
            if os.path.exists(self.audio_log_file) and os.path.getsize(self.audio_log_file) > 0:
                with open(self.audio_log_file, 'r') as f:
                    try:
                        current_logs = json.load(f)
                        if not isinstance(current_logs, list): current_logs = []
                    except json.JSONDecodeError: current_logs = []
            current_logs.append(log_entry)
            with open(self.audio_log_file, 'w') as f:
                json.dump(current_logs, f, indent=4)
        except IOError as e: print(f"AUDIO_AGENT: Error writing to audio log: {e}")
        except Exception as e: print(f"AUDIO_AGENT: Unexpected error during audio logging: {e}")

    def update_camera_context(self,context: str):
        self.camera_context = context
        print(f"AUDIO_AGENT: Camera context manually updated: {context[:100]}...")

    def refresh_camera_context_from_log(self, max_entries=3):
        try:
            if not os.path.exists(self.camera_log_file) or os.path.getsize(self.camera_log_file) == 0:
                return
            with open(self.camera_log_file, 'r') as f:
                try:
                    all_camera_logs = json.load(f)
                    if not isinstance(all_camera_logs, list): all_camera_logs = []
                except json.JSONDecodeError:
                    return
            if not all_camera_logs: return

            recent_logs = all_camera_logs[-max_entries:]
            relevant_comments = [log.get("comment_by_llm", "").strip() for log in recent_logs if isinstance(log.get("comment_by_llm"), str)]
            relevant_comments = [comment for comment in relevant_comments if comment]

            if relevant_comments:
                new_context = " ".join(relevant_comments)
                if new_context != self.camera_context:
                    self.camera_context = new_context
                    print(f"AUDIO_AGENT: Camera context refreshed: {self.camera_context[:200]}...")
        except IOError as e: print(f"AUDIO_AGENT: Error reading camera log: {e}")
        except Exception as e: print(f"AUDIO_AGENT: Unexpected error refreshing camera context: {e}")

    async def start_adk_session(self, user_id: str, is_audio: bool = True):
        if user_id in self.active_adk_sessions:
            print(f"AUDIO_AGENT: Session for user {user_id} already exists. Closing old one.")
            await self.stop_adk_session(user_id)

        self.refresh_camera_context_from_log()

        # Dynamically create the agent's instruction with the latest camera context.
        current_instruction = (
            self.base_agent_instruction +
            f" Current visual context from surroundings: '{self.camera_context}'."
        )

        # Create a new ADK Agent instance with the updated instruction for this session.
        temp_agent_for_session = Agent(
           name=f'streaming_audio_root_agent_{user_id}', # Unique name per session
           model=self.model_name,
           description=self.base_agent_description,
           instruction=current_instruction
        )

        print(f"AUDIO_AGENT: Starting ADK session for {user_id}, audio: {is_audio}")
        try:
            runner = InMemoryRunner(app_name=f'{APP_NAME}_{user_id}', agent=temp_agent_for_session)

            session = await runner.session_service.create_session(
                app_name=f'{APP_NAME}_{user_id}',
                user_id=user_id
            )
            modality = "AUDIO" if is_audio else "TEXT"
            run_config = RunConfig(response_modalities=[modality])

            live_request_queue = LiveRequestQueue()

            # No INITIAL REQUEST to run_live. The agent should wait for user input.
            # The context is now part of the agent's instruction for this session.
            live_events = runner.run_live(
                session=session,
                live_request_queue=live_request_queue,
                run_config=run_config,
                # request=None  -- DO NOT SEND AN INITIAL TEXT REQUEST TO LIVE AUDIO MODELS
            )

            self.active_adk_sessions[user_id] = {
                "runner": runner,
                "session": session,
                "live_request_queue": live_request_queue,
                "live_events": live_events,
                "is_audio": is_audio
            }
            print(f"AUDIO_AGENT: ADK session started for {user_id} with context: '{self.camera_context[:100]}...'")
            return live_events, live_request_queue
        except Exception as e:
            print(f'AUDIO_AGENT: Error starting ADK session for {user_id}: {e}')
            if user_id in self.active_adk_sessions: del self.active_adk_sessions[user_id]
            raise

    async def stop_adk_session(self, user_id: str):
        if user_id in self.active_adk_sessions:
            print(f"AUDIO_AGENT: Stopping ADK session for {user_id}...")
            self.active_adk_sessions[user_id]["live_request_queue"].close()
            del self.active_adk_sessions[user_id]
            print(f"AUDIO_AGENT: ADK session stopped for {user_id}.")

    async def agent_to_client_sse_handler(self, user_id: str):
        if user_id not in self.active_adk_sessions:
            print(f"AUDIO_AGENT SSE: Session not found for user {user_id}.")
            yield f"data: {json.dumps({'error': 'Session not found.'})}\n\n"; return

        live_events = self.active_adk_sessions[user_id]["live_events"]
        is_audio = self.active_adk_sessions[user_id]["is_audio"]

        text_turn = ""

        print(f"AUDIO_AGENT SSE: Starting event stream for user {user_id}.")
        async for event in live_events:
            if event.turn_complete or event.interrupted:
                message = {"turn_complete": event.turn_complete,"interrupted": event.interrupted}
                yield f"data: {json.dumps(message)}\n\n"
                if text_turn: # Log any accumulated text from the agent
                    self._log_transcription("agent", text_turn)
                text_turn = ""
                continue

            part = event.content and event.content.parts and event.content.parts[0]
            if not part: continue

            if is_audio and part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                if part.inline_data.data:
                    yield f"data: {json.dumps({'mime_type': part.inline_data.mime_type, 'data': base64.b64encode(part.inline_data.data).decode('ascii')})}\n\n"

            if part.text:
                yield f"data: {json.dumps({'mime_type': 'text/plain', 'data': part.text})}\n\n"
                text_turn += part.text

    async def client_to_agent_handler(self, user_id: str, client_msg: dict):
        if user_id not in self.active_adk_sessions:
            print(f"AUDIO_AGENT MSG: Session not found for user {user_id} in client_to_agent_handler.")
            return {"error": "Session not found."}

        queue = self.active_adk_sessions[user_id]["live_request_queue"]
        is_audio = self.active_adk_sessions[user_id]["is_audio"]
        mime, data = client_msg.get("mime_type"), client_msg.get("data")

        if not mime or data is None: return {"error": "Invalid message."}

        log_text = ""
        if mime == "text/plain":
            log_text = data
            queue.send_content(Content(role="user", parts=[Part.from_text(data)]))
            print(f"AUDIO_AGENT MSG [CLIENT->AGENT] ({user_id}): Text: '{data[:50]}...'")
        elif mime.startswith('audio/') and is_audio:
            try:
                decoded = base64.b64decode(data)
                queue.send_realtime(Blob(data=decoded, mime_type=mime))
                print(f"AUDIO_AGENT MSG [CLIENT->AGENT] ({user_id}): {mime} {len(decoded)} bytes.")
            except Exception as e: return {"error": f"Audio error: {e}"}
        else: return {"error": "Unsupported mime/session."}

        if log_text:
            self._log_transcription("user", log_text)

        return {"status": "sent"}

if __name__ == '__main__':
    print("AUDIO_AGENT: Conceptual test.")
    project, loc = os.getenv('GOOGLE_CLOUD_PROJECT'), os.getenv('GOOGLE_CLOUD_LOCATION')
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI','F').lower()=='t' and project and loc:
        agent = AudioAgent(project, loc)
        with open(agent.camera_log_file, 'w') as f: json.dump([{"comment_by_llm": "Red car passed."},{"comment_by_llm": "Dog barked."}], f, indent=2) # Corrected: indent=2
        print(f"Initial ctx: '{agent.camera_context}'")
        agent.refresh_camera_context_from_log()
        print(f"Refreshed ctx: '{agent.camera_context}'")
        async def main_test():
            uid = "test_uid_004"
            try:
                await agent.start_adk_session(uid, is_audio=False)
                await agent.client_to_agent_handler(uid, {"mime_type": "text/plain", "data": "What's new?"})
            finally: await agent.stop_adk_session(uid)
        asyncio.run(main_test())
        if os.path.exists(agent.camera_log_file): os.remove(agent.camera_log_file)
        print("AUDIO_AGENT: Test done.")
    else: print("CAUDIO_AGENT: Skip test, Vertex AI env not set.")
