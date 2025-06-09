# Import necessary Python modules
import os  # For interacting with the operating system (e.g., environment variables, file paths)
import asyncio  # For asynchronous programming, core to FastAPI's async features
import uvicorn  # The ASGI server that will run the FastAPI application; it's used here
                #  if the script is run directly, but typically you run Uvicorn from
                #  the command line.

# Third-party libraries for FastAPI
from fastapi import FastAPI, Request, HTTPException  # Core FastAPI classes, Request object for incoming requests, & HTTPException for errors.
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse  # Different types of HTTP responses.
from fastapi.staticfiles import StaticFiles  # For serving static files (that is, files that don't change, like HTML, CSS, JS).
from fastapi.middleware.cors import CORSMiddleware  # For handling Cross-Origin Resource Sharing (commonly needed for web apps).

# Python standard libraries
from pathlib import Path  # For object-oriented manipulation of file system paths.
import json  # For parsing JSON data.

# Application-specific imports from this project
# The `.` in `.camera_agent` means that the module is in the same directory ("app") as this file.
from .camera_agent import CameraAgent  # Import the CameraAgent class.
from .audio_agent import AudioAgent  # Import the AudioAgent class.

# Import for loading environment variables from a `.env` file.
from dotenv import load_dotenv

# --- Configuration and Environment Setup ---
# Define the path to the `.env` file.
# __file___ represents the path to the current file (i.e., `streaming-test2/app/main.py`).
# Path(__file__).resolve() ensures an absolute path.
  # .parent takes us from `app/main.py` to `app/`.
  # .parent.parent takes us from `app/` to `streaming-test2/`.
  # Then we append '.env' to form the full path.
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)  # Load the environment variables from the .env file.

# Read environment variables after loading the .env file.
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# --- FastAPI Application Initialization ---
# Create an instance of the FastAPI class. This instance will be the core
# of the web application, handling all incoming requests and routing them to
# the appropriate handler functions.
app = FastAPI()

# CORS (Cross-Origin Resource Sharing) Middleware
# CORS is a security feature in web browsers that restricts how a web page
# from one origin (domain) can make requests to a different origin.
# For development, it's common to allow all origins ("*") to make things easier.
# In a production environment, you should restrict this only to the allowed
# origins of your frontend application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin.
    allow_credentials=True,  # Allows cookies and authorization headers to be sent.
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, delete, etc.).
    allow_headers=["*"],  # Allows all HTTP headers in requests.
)

# Define the path to the directory containing static files (HTML, CSS< JS).
#  This path is relative to the location of this main.py file.
STATIC_DIR = Path(__file__).resolve().parent / "static"  # Path is `streaming-test2/app/static/`
# Mount the static directory. This tells FastAPI that any request that starts
# with `/static` should be served from the directory specified by STATIC_DIR.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Define the path to the logs directory.
# This path is relative to the location of this main.py file.
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"  # Path is `streaming-test2/logs/`

# Global variables to hold the singleton instances of the agents.
# They are initialized to None and will get set during the `startup_event`.
camera_agent_instance = None
audio_agent_instance = None

# --- FastAPI Event Handlers ---
@app.on_event("startup")  # This decorator tells FastAPI to run the `startup_event` function
                          #  once when the server starts up.
async def startup_event():
    global camera_agent_instance, audio_agent_instance  # Declare that we will modify the global variables.
    print("MAIN_APP: FastAPI server starting up...")

    # Critical check for environment variables. If these are not set, the agents
    #  will not be able to function correctly.
    if not PROJECT_ID or not LOCATION:
        print(f"MAIN_APP: FATAL - GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set in .env (path: {dotenv_path}). Agents will not initialize.")
        return  # Stop further startup process if config is missing.

    try:
        print("MAIN_APP: Initializing CameraAgent...")
        # Create an instance of the CameraAgent.
        camera_agent_instance = CameraAgent(project_id=PROJECT_ID, location=LOCATION)
        # Start the CameraAgent's monitoring loop, which runs in a separate thread.
        camera_agent_instance.start()
        print("MAIN_APP: CameraAgent initialized and started.")
    except Exception as e:
        print(f"MAIN_APP: Error initializing CameraAgent: {e}")
        camera_agent_instance = None  # If initialization fails, ensure the instance is None.

    try:
        print("MAIN_APP: Initializing AudioAgent...")
        # Create an instance of the AudioAgent.
        audio_agent_instance = AudioAgent(project_id=PROJECT_ID, location=LOCATION)
        print("MAIN_APP: AudioAgent initialized.")
    except Exception as e:
        print(f"MAIN_APP: Error initializing AudioAgent: {e}")
        audio_agent_instance = None  # If initialization fails, ensure the instance is None.

@app.on_event("shutdown")  # This decorator tells FastAPI to run the `shutdown_event` function once when the
                           #  server is shutting down.
async def shutdown_event():
    print("MAIN_APP: FastAPI server shutting down...")
    if camera_agent_instance:
        print("MAIN_APP: Stopping CameraAgent...")
        # Call the `stop` method of the CameraAgent to cleanly stop its background thread.
        camera_agent_instance.stop()

    # Audio agent sessions are managed by the AudioAgent itself, typically when
    # an SSE connection closes. No explicit action for the AudioAgent itself is
    # needed here in the global shutdown handler, unless there's a global cleanup
    # for the AudioAgent object itself.


# --- HTML Page Endpoint ---
@app.get("/", response_class=HTMLResponse)  # Defines a route for the root URL ("/").
                                            # IT  expects to return an HTMLResponse.
async def get_root():
    # This asynchronous function will be called when a web browser makes a GET request to the root page.
    # It returns the `index.html` file from the `STATIC_DIR` that was defined earlier.
    return FileResponse(STATIC_DIR / "index.html")


# --- Audio Agent Endpoints ---
@app.get("/events/{user_id}")  # Defines an SSE endpoint. {user_id} is a path parameter.
async def audio_sse_endpoint(user_id: str, request: Request, is_audio: str = "true"):
    """
    SSE (Server-Sent Events) endpoint for the AudioAgent to stream events to the client.
    When a client connects to this endpoint, an ADK audio session is started for the user.
    Events (e.g., audio chunks, text transcriptions, turn completion notices) are
    sent to the client over this persistent HTTP connection.
    """
    # Check if the audio agent instance has been initialized.
    if not audio_agent_instance:
        print("MAIN_APP SSE: AudioAgent not initialized. Cannot start session.")
        raise HTTPException(status_code=503, detail="AudioAgent not available.")

    print(f"MAIN_APP SSE request for user: {user_id}, audio: {is_audio}")
    try:
        # Start an ADK session for this user. This is an async function in the AudioAgent.
        # It returns a tuple of the `live_events` async generator and the live request queue.
        # is_audio is passed as a boolean to the AudioAgent.
        live_events, _ = await audio_agent_instance.start_adk_session(user_id, is_audio == "true")

        async def event_generator():
            """
Async generator that yields data chunks from the AudioAgent's SSE processor.
It also monitors if the client disconnects, and performs cleanup.
            """
            try:
                # Iterate over the `live_events` stream provided by the AudioAgent.
                async for data_chunk in audio_agent_instance.agent_to_client_sse_handler(user_id):
                    # if the client disconnects during streaming, stop sending events.
                    if await request.is_disconnected():
                        break
                    yield data_chunk  # Yield the data chunk to the client.
            except asyncio.CancelledError:  # Handles cases where the client disconnects abruptly.
                print(f"MAIN_APP: SSE for {user_id} cancelled.")
            except Exception as e:
                print(f"MAIN_APP: SSE error for {user_id}: {e}")
            finally:
                # This block executes regardless of how the try block exited.
                # It's used to ensure that the ADK session is properly closed and cleaned up.
                print(f"MAIN_APP: Cleaning ADK session for {user_id} (SSE ended).")
                await audio_agent_instance.stop_adk_session(user_id)

        # Return a StreamingResponse, which is FastAPI's way to handle SSE.
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        print(f"MAIN_APP: Error starting SSE for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f'Failed to start session: {str(e)}')

@app.post("/send/{user_id}")  # Defines a POST route to handle data sent by the client.
async def audio_send_message_endpoint(user_id: str, request: Request):
    """
    HTTP endpoint for the client to send data (text or audio) to the AudioAgent.
    It acts as a bridge, passing the request data to the AudioAgent's `client_to_agent_handler`
    method for processing.
    """
    if not audio_agent_instance:
        print("MAIN_APP SEND ERROR: AudioAgent not initialized.")
        raise HTTPException(status_code=503, detail="AudioAgent not available.")

    # Parse the JSON payload from the incoming request.
    try:
        message_data = await request.json()
    except json.JSONDecodeError:
        print("MAIN_APP SEND ERROR: Invalid JSON received.")
        raise HTTPException(status_code=400, detail="Invalid JSON.")

    # Delegate the processing of the message to the AudioAgent's dedicated handler.
    response = await audio_agent_instance.client_to_agent_handler(user_id, message_data)

    # If the AudioAgent returns an error, reflect that in the HTTP response.
    if response.get("error"):
        print(f"MAIN_APP SEND ERROR: Agent returned error for user {user_id}: {response['error']}")
        raise HTTPException(status_code=400, detail=response["error"])
    return response

# --- Log File Fetching Endpoints ---
@app.get("/logs/{log_type}")  # Defines a route to allow the frontend to fetch log files.
async def get_log_content(log_type: str):
    """
    Provides an endpoint to retrieve the contents of the specified log file
    (`camera_log.json` or `audio_log.json`).
    This allows the frontend to periodically update the log displays for the user.
    """
    # Validate the requested log type.
    if log_type not in ["camera", "audio"]:
        print(f"MAIN_APP LOG ERROR: Invalid log type requested: {log_type}")
        raise HTTPException(status_code=404, detail="Invalid log type. Use 'camera' or 'audio'.")

    # Construct the full path to the requested log file.
    log_file_name = f"{log_type}_log.json"
    log_file_path = LOGS_DIR / log_file_name

    # If the log file doesn't exist or is empty, return an empty JSON list
    # to prevent errors on the client side.
    if not log_file_path.exists() or log_file_path.stat().st_size == 0:
        return JSONResponse(content=[], status_code=200)

    try:
        # Open the log file in read mode and load its JSON content.
        with open(log_file_path, 'r') as f:
            log_content = json.load(f)
        return JSONResponse(content=log_content)
    except Exception as e:  # Catch JSONDecodeError and other potential errors.
        print(f"MAIN_APP LOG ERROR: Error reading log file {log_file_path}: {e}")
        # Return a JSON response containing an error message. This helps the client
        # to handle the error gracefully instead of getting a broken 500 error.
        return JSONResponse(content=[{"error": f"Failed to parse {log_file_name}"}], status_code=200)


@app.get("/agent/camera/status")  # Endpoint to get the CameraAgent's current status
async def get_camera_agent_status():
    """
    Returns the current status of the CameraAgent, including whether it is
    running and if the video capture is active.
    """
    if camera_agent_instance and camera_agent_instance.running:
        return {"status": "running", "monitoring": (camera_agent_instance.video_monitor.cap is not None) and camera_agent_instance.video_monitor.cap.isOpened()}
    return {"status": "stopped"}

# --- Run the Application (for local development) ---
if __name__ == "__main__":
    # This block is executed when the script is run directly (e.g., `python main.py`).
    # It's a common Python idiom for code that should only run when the module
    # is the 'main' program, not when it's imported as module into another script.

    print("MAIN_APP: To run this app, use Uvicorn from the command line. Example:")
    print("uvicorn streaming-test2.app.main:app --reload --port 8000 --host 0.0.0.0 --app-dir .  (from the repository root)")

    # The following `uvicorn.run()` is commented out as it's generally better to run
    # Uvicorn from the command line to hate full control over its options and
    # for better integration with process managers.
    # If you were to run this file directly with `python streaming-test2/app/main.py`
    # from the repository root, and you uncommented the below,
    # you would need `app_dir="app"` to tell Uvicorn where to find `main:app`
    # (i.e., the `app` object in the `main` file), given the execution path.
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="app")
