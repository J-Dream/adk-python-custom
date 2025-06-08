import os
import json
import base64
import datetime
import warnings
import asyncio # Required for async operations
from pathlib import Path
from dotenv import load_dotenv

from google.genai.types import Part, Content, Blob
from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue, Agent
from google.adk.agents.run_config import RunConfig

# Load environment variables from .env file in the root of streaming-test2
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "ADK_Streaming_Audio_Agent"

class AudioAgent:
    def __init__(self, project_id: str, location: str, model_name: str = "gemini-2.0-flash-live-preview-04-09"):
        if not project_id:
            raise ValueError("AUDIO_AGENT: Google Cloud Project ID is required.")
        if not location:
            raise ValueError("AUDIO_AGENT: Google Cloud Location is required.")

        self.project_id = project_id
        self.location = location
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() != "true":
            raise EnvironmentError("AUDIO_AGENT: GOOGLE_GENAI_USE_VERTEXAI must be set to True in .env")

        self.root_agent_adk = Agent(
           name="streaming_audio_root_agent",
           model=model_name,
           description="Live audio conversational agent.",
           instruction="You are a helpful voice assistant. Respond to the user's speech. Be concise and natural. When appropriate, consider the following visual context from your surroundings.",
        )
        print(f"AUDIO_AGENT: Initialized with ADK root agent using model: {model_name}")

        self.audio_log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'audio_log.json')
        self.camera_log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'camera_log.json')

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

    def _log_transcription(self, speaker: str, text: str):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, "speaker": speaker, "text": text}
        try:
            current_logs = []
            if os.path.exists(self.audio_log_file) and os.path.getsize(self.audio_log_file) > 0: # Check if file exists and is not empty
                with open(self.audio_log_file, 'r') as f:
                    try:
                        current_logs = json.load(f)
                        if not isinstance(current_logs, list): current_logs = []
                    except json.JSONDecodeError: current_logs = [] # If file is corrupt, start fresh
            current_logs.append(log_entry)
            with open(self.audio_log_file, 'w') as f:
                json.dump(current_logs, f, indent=4)
        except IOError as e: print(f"AUDIO_AGENT: Error writing to audio log: {e}")
        except Exception as e: print(f"AUDIO_AGENT: Unexpected error during audio logging: {e}")

    def update_camera_context(self, context: str):
        self.camera_context = context
        print(f"AUDIO_AGENT: Camera context manually updated: {context[:100]}...")

    def refresh_camera_context_from_log(self, max_entries=3):
        try:
            if not os.path.exists(self.camera_log_file) or os.path.getsize(self.camera_log_file) == 0:
                # print(f"AUDIO_AGENT: Camera log missing or empty at {self.camera_log_file}.") # Can be noisy
                return
            with open(self.camera_log_file, 'r') as f:
                try:
                    all_camera_logs = json.load(f)
                    if not isinstance(all_camera_logs, list): all_camera_logs = []
                except json.JSONDecodeError:
                    # print(f"AUDIO_AGENT: Could not decode camera log: {self.camera_log_file}.") # Can be noisy
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
        if user_id in self.active_adk_sessions: await self.stop_adk_session(user_id)
        self.refresh_camera_context_from_log()
        print(f"AUDIO_AGENT: Starting ADK session for {user_id}, audio: {is_audio}")
        try:
            runner = InMemoryRunner(app_name=f"{APP_NAME}_{user_id}", agent=self.root_agent_adk)
            session = await runner.session_service.create_session(app_name=f"{APP_NAME}_{user_id}", user_id=user_id)
            modality = "AUDIO" if is_audio else "TEXT"
            run_config = RunConfig(response_modalities=[modality])
            live_request_queue = LiveRequestQueue()
            initial_msg = (f"System Info: Visual context: '{self.camera_context}'. Please wait for user.")
            initial_content = Content(role="user", parts=[Part.from_text(initial_msg)])
            live_events = runner.run_live(session=session, live_request_queue=live_request_queue, run_config=run_config, request=initial_content)
            self.active_adk_sessions[user_id] = {"runner": runner, "session": session, "live_request_queue": live_request_queue, "live_events": live_events, "is_audio": is_audio}
            print(f"AUDIO_AGENT: ADK session started for {user_id} with context: {self.camera_context[:100]}...")
            return live_events, live_request_queue
        except Exception as e:
            print(f"AUDIO_AGENT: Error starting ADK session for {user_id}: {e}")
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
            yield f"data: {json.dumps({'error': 'Session not found.'})}

"; return
        live_events = self.active_adk_sessions[user_id]["live_events"]
        is_audio = self.active_adk_sessions[user_id]["is_audio"]
        text_turn = ""
        async for event in live_events:
            if event.turn_complete or event.interrupted:
                yield f"data: {json.dumps({'turn_complete': event.turn_complete, 'interrupted': event.interrupted})}

"
                if text_turn and "System Info:" not in text_turn: self._log_transcription("agent", text_turn)
                text_turn = ""
                continue
            part = event.content and event.content.parts and event.content.parts[0]
            if not part: continue
            if is_audio and part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                if part.inline_data.data: yield f"data: {json.dumps({'mime_type': part.inline_data.mime_type, 'data': base64.b64encode(part.inline_data.data).decode('ascii')})}

"
            if part.text:
                if "System Info:" in part.text and event.content.role == "model": continue
                yield f"data: {json.dumps({'mime_type': 'text/plain', 'data': part.text})}

"
                text_turn += part.text # Accumulate all text parts for the turn
        # print(f"AUDIO_AGENT SSE: Stream ended for {user_id}.") # Can be noisy

    async def client_to_agent_handler(self, user_id: str, client_msg: dict):
        if user_id not in self.active_adk_sessions: return {"error": "Session not found."}
        # self.refresh_camera_context_from_log() # Context is primarily set at session start
        queue = self.active_adk_sessions[user_id]["live_request_queue"]
        is_audio = self.active_adk_sessions[user_id]["is_audio"]
        mime, data = client_msg.get("mime_type"), client_msg.get("data")
        if not mime or data is None: return {"error": "Invalid message."}
        log_text = ""
        if mime == "text/plain":
            log_text = data
            queue.send_content(Content(role="user", parts=[Part.from_text(data)]))
            print(f"AUDIO_AGENT MSG [CLIENT->AGENT] ({user_id}): Text: '{data[:50]}...'")
        elif mime.startswith("audio/") and is_audio:
            try:
                decoded = base64.b64decode(data)
                queue.send_realtime(Blob(data=decoded, mime_type=mime))
                print(f"AUDIO_AGENT MSG [CLIENT->AGENT] ({user_id}): {mime} {len(decoded)} bytes.")
            except Exception as e: return {"error": f"Audio error: {e}"}
        else: return {"error": "Unsupported mime/session."}
        if log_text: self._log_transcription("user", log_text)
        return {"status": "sent"}

if __name__ == '__main__':
    print("AUDIO_AGENT: Conceptual test.")
    project, loc = os.getenv('GOOGLE_CLOUD_PROJECT'), os.getenv('GOOGLE_CLOUD_LOCATION')
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI','F').lower()=='t' and project and loc:
        agent = AudioAgent(project, loc)
        dummy_cam_log = agent.camera_log_file
        with open(dummy_cam_log, 'w') as f: json.dump([{"comment_by_llm": "Red car passed."},{"comment_by_llm": "Dog barked."}], f)
        print(f"Initial ctx: '{agent.camera_context}'")
        agent.refresh_camera_context_from_log()
        print(f"Refreshed ctx: '{agent.camera_context}'") # Should be "Red car... Dog barked."
        async def main_test():
            uid = "test_uid_004"
            try:
                await agent.start_adk_session(uid, is_audio=False)
                await agent.client_to_agent_handler(uid, {"mime_type": "text/plain", "data": "What's new?"})
            finally: await agent.stop_adk_session(uid)
        asyncio.run(main_test())
        if os.path.exists(dummy_cam_log): os.remove(dummy_cam_log)
        print("AUDIO_AGENT: Test done.")
    else: print("AUDIO_AGENT: Skip test, Vertex AI env not set.")
