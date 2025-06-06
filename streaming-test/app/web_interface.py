from flask import Flask, render_template, Response, request, jsonify, stream_with_context
import time
import json # For SSE
from app.agent import MultimodalAgent # Assuming agent is in app.agent
from app.config import WEB_HOST, WEB_PORT, MODEL_NAME # Assuming these are in config
import os
from dotenv import load_dotenv
import queue # For SSE messages

# --- Global Agent Initialization ---
# Load .env file from the parent directory of 'app'
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

agent_instance = None
agent_event_queue = queue.Queue() # Queue for SSE messages from agent

def get_agent():
    global agent_instance
    if agent_instance is None:
        if use_vertex and project_id and location:
            print(f"Initializing Agent for web interface with project: {project_id}, loc: {location}")
            agent_instance = MultimodalAgent(
                project_id=project_id,
                location=location,
                model_name=MODEL_NAME, # from app.config
                camera_index=0 # Default camera
            )
            # We need a way for the agent to push updates (like video change comments)
            # One way is to pass a callback or a queue to the agent.
            # For now, the web routes will call agent methods.
            # For SSE, the agent could put messages onto agent_event_queue.
        else:
            print("Web Interface: Agent cannot be initialized. Check .env settings.")
            # Raise an error or handle gracefully in routes
    return agent_instance

# --- Flask App Setup ---
web_app = Flask(__name__) # Renamed from app to web_app to avoid conflict if 'app' is a module

def generate_video_frames():
    local_agent = get_agent()
    if not local_agent or not local_agent.video_monitor:
        print("Video frames: Agent or video monitor not available.")
        # Return a placeholder image or error
        # For now, just yield nothing if not available, or handle error
        return

    if not local_agent.video_monitor.cap or not local_agent.video_monitor.cap.isOpened():
        if not local_agent.video_monitor.start_capture():
            print("Video frames: Failed to start camera for web feed.")
            return # Exit generator if camera fails

    while True:
        if not local_agent.video_monitor.cap or not local_agent.video_monitor.cap.isOpened():
            print("Video stream: camera became unavailable.")
            break # Stop if camera is no longer available

        frame = local_agent.video_monitor.get_frame()
        if frame is None:
            # print("Video stream: failed to get frame.")
            # Could yield a "no signal" image here
            time.sleep(0.1) # Avoid busy loop if frames stop
            continue

        # Encode frame as JPEG
        ret, buffer = local_agent.video_monitor.cv2.imencode('.jpg', frame)
        if not ret:
            # print("Video stream: failed to encode frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Check for video changes and send to model (and potentially SSE queue)
        # This runs video processing in the web request thread for /video_feed
        # which might not be ideal for performance.
        # A background thread for the agent's video processing loop is better.
        # For now, let's call the agent's check method.
        # Consider that check_for_video_changes_and_comment might call the model, which is slow.
        # This is a simplified integration.

        # Decouple the model call from the video feed generation
        # The agent's own loop (if we create one later in main.py) should handle this.
        # For now, we will NOT call agent.check_for_video_changes_and_comment() here
        # to keep /video_feed responsive. Video change events should come via SSE from a background process.

        time.sleep(1.0 / local_agent.video_monitor.fps_limit if local_agent.video_monitor.fps_limit else 0.03)


@web_app.route('/')
def index():
    return render_template('index.html')

@web_app.route('/video_feed')
def video_feed():
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@web_app.route('/start_interaction', methods=['GET'])
def start_interaction():
    local_agent = get_agent()
    if not local_agent:
        return jsonify({"error": "Agent not initialized. Check server logs and .env settings."}), 500
    try:
        local_agent.start_chat() # This also starts video capture
        # Add a status message to SSE queue
        agent_event_queue.put({"type": "status", "message": "Agent started. Video capture active."})
        return jsonify({"status": "Agent interaction started. Video capture active."})
    except Exception as e:
        print(f"Error starting agent interaction: {e}")
        return jsonify({"error": str(e)}), 500

@web_app.route('/stop_interaction', methods=['GET'])
def stop_interaction():
    local_agent = get_agent()
    if not local_agent:
        return jsonify({"error": "Agent not initialized."}), 500
    try:
        local_agent.stop_chat() # This also stops video capture
        agent_event_queue.put({"type": "status", "message": "Agent stopped. Video capture released."})
        return jsonify({"status": "Agent interaction stopped."})
    except Exception as e:
        print(f"Error stopping agent interaction: {e}")
        return jsonify({"error": str(e)}), 500

@web_app.route('/send_chat_message', methods=['POST'])
def send_chat_message():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    local_agent = get_agent()
    if not local_agent:
        return jsonify({"error": "Agent not available."}), 503

    if not local_agent.chat: # Ensure chat is started
        local_agent.start_chat()

    try:
        # For now, web chat sends text, gets text. Voice interaction is separate.
        model_reply = local_agent.send_text_message(user_message)
        # Put user message and model reply onto the event queue for SSE clients too if desired
        # agent_event_queue.put({"type": "user_message_web", "text": user_message})
        # agent_event_queue.put({"type": "model_response_web", "text": model_reply})
        return jsonify({"reply": model_reply})
    except Exception as e:
        print(f"Error processing chat message: {e}")
        return jsonify({"error": str(e)}), 500

# Server-Sent Events endpoint
def generate_agent_events():
    # This function will yield messages from the agent_event_queue
    # The agent's background tasks (audio processing, video change detection)
    # should put messages (dict or JSON string) into this queue.
    print("SSE client connected. Streaming events...")
    try:
        while True:
            message = agent_event_queue.get() # Blocking call
            # print(f"SSE: Sending event: {message}") # For debugging
            yield f"data: {json.dumps(message)}\n\n"
            agent_event_queue.task_done() # Mark task as done
            time.sleep(0.1) # Short delay to allow multiple messages to batch if rapidly produced
    except GeneratorExit:
        print("SSE client disconnected.")
    except Exception as e:
        print(f"Error in SSE event generator: {e}")


@web_app.route('/agent_events')
def agent_events():
    return Response(stream_with_context(generate_agent_events()), mimetype="text/event-stream")


# The main execution block for running Flask app directly from this file
# This will be superseded by run.py or main.py starting the web server.
if __name__ == '__main__':
    print("Starting Flask web server directly from web_interface.py (for testing)...")
    print(f"Agent use_vertex: {use_vertex}, project: {project_id}, location: {location}")
    if not (use_vertex and project_id and location):
        print("WARNING: Agent will not be fully functional due to missing .env vars.")

    # Example of how agent's background task could put messages to queue:
    # This needs to be run in a separate thread in a real app.
    def dummy_agent_activity_simulator():
        count = 0
        ag = get_agent() # Ensure agent is initialized
        if ag and not ag.chat : ag.start_chat() # Start agent if not already

        while True:
            time.sleep(5)
            # Simulate a video change event
            if ag and ag.video_monitor and ag.video_monitor.cap and ag.video_monitor.cap.isOpened():
                 changed, desc, frame_bytes = ag.video_monitor.process_frame_for_changes()
                 if changed and frame_bytes:
                     print(f"Background check: Video change detected: {desc}")
                     # In a real scenario, the agent's own method would call the model
                     # and then put the model's comment on the queue.
                     # For now, just put the description.
                     # This call to process_frame_for_changes might be redundant if /video_feed is active
                     # and also calling it. A proper background thread is needed.
                     # For now, this is just a conceptual test for SSE.

                     # Simplified: Let's assume agent.check_for_video_changes_and_comment()
                     # is modified to put its findings on the queue.
                     # For this test, we'll manually put a message.
                     comment_from_model = f"Model observes: {desc} (event {count})"
                     agent_event_queue.put({
                         "type": "video_change_comment",
                         "comment": comment_from_model,
                         # "image_url": "/last_detected_change.jpg" # If we save the image
                     })
                     print(f"SSE Queue: Put video change: {comment_from_model}")

            count += 1
            # Simulate a transcription event (if agent was handling voice)
            # agent_event_queue.put({"type": "transcription", "text": f"User said something {count}"})
            # Simulate model audio response text
            # agent_event_queue.put({"type": "model_response_audio_text", "text": f"Agent replied something {count}"})


    # import threading
    # simulator_thread = threading.Thread(target=dummy_agent_activity_simulator, daemon=True)
    # simulator_thread.start()
    # print("Dummy agent activity simulator thread started for SSE testing.")

    web_app.run(host=WEB_HOST, port=WEB_PORT, debug=True, threaded=True, use_reloader=False)
    # use_reloader=False because it can cause issues with global state like agent_instance
    # and background threads. For development, one might enable it but be wary of agent re-initialization.
    # threaded=True is important for handling multiple requests like /video_feed and /agent_events concurrently.
