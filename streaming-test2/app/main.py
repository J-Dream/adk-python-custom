import os
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import json

from .camera_agent import CameraAgent
from .audio_agent import AudioAgent

from dotenv import load_dotenv
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

camera_agent_instance = None
audio_agent_instance = None

@app.on_event("startup")
async def startup_event():
    global camera_agent_instance, audio_agent_instance
    print("MAIN_APP: FastAPI server starting up...")
    if not PROJECT_ID or not LOCATION:
        print(f"MAIN_APP: FATAL - GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION not set in .env (path: {dotenv_path}). Agents will not initialize.")
        return

    try:
        print("MAIN_APP: Initializing CameraAgent...")
        camera_agent_instance = CameraAgent(project_id=PROJECT_ID, location=LOCATION)
        camera_agent_instance.start()
        print("MAIN_APP: CameraAgent initialized and started.")
    except Exception as e:
        print(f"MAIN_APP: Error initializing CameraAgent: {e}")
        camera_agent_instance = None

    try:
        print("MAIN_APP: Initializing AudioAgent...")
        audio_agent_instance = AudioAgent(project_id=PROJECT_ID, location=LOCATION)
        print("MAIN_APP: AudioAgent initialized.")
    except Exception as e:
        print(f"MAIN_APP: Error initializing AudioAgent: {e}")
        audio_agent_instance = None

@app.on_event("shutdown")
async def shutdown_event():
    print("MAIN_APP: FastAPI server shutting down...")
    if camera_agent_instance:
        print("MAIN_APP: Stopping CameraAgent...")
        camera_agent_instance.stop()

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/events/{user_id}")
async def audio_sse_endpoint(user_id: str, request: Request, is_audio: str = "true"):
    if not audio_agent_instance: raise HTTPException(status_code=503, detail="AudioAgent unavailable.")
    print(f"MAIN_APP: SSE request for user: {user_id}, audio: {is_audio}")
    try:
        live_events, _ = await audio_agent_instance.start_adk_session(user_id, is_audio == "true")
        async def event_generator():
            try:
                async for data_chunk in audio_agent_instance.agent_to_client_sse_handler(user_id):
                    if await request.is_disconnected(): break
                    yield data_chunk
            except asyncio.CancelledError: print(f"MAIN_APP: SSE for {user_id} cancelled.")
            except Exception as e: print(f"MAIN_APP: SSE error for {user_id}: {e}")
            finally:
                print(f"MAIN_APP: Cleaning ADK session for {user_id} (SSE ended).")
                await audio_agent_instance.stop_adk_session(user_id)
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        print(f"MAIN_APP: Error starting SSE for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {e}")

@app.post("/send/{user_id}")
async def audio_send_message_endpoint(user_id: str, request: Request):
    if not audio_agent_instance: raise HTTPException(status_code=503, detail="AudioAgent unavailable.")
    try: message_data = await request.json()
    except json.JSONDecodeError: raise HTTPException(status_code=400, detail="Invalid JSON.")
    response = await audio_agent_instance.client_to_agent_handler(user_id, message_data)
    if response.get("error"): raise HTTPException(status_code=400, detail=response["error"])
    return response

@app.get("/logs/{log_type}")
async def get_log_content(log_type: str):
    if log_type not in ["camera", "audio"]: raise HTTPException(status_code=404, detail="Invalid log type.")
    log_file = LOGS_DIR / f"{log_type}_log.json"
    if not log_file.exists() or log_file.stat().st_size == 0: return JSONResponse(content=[])
    try:
        with open(log_file, 'r') as f: log_content = json.load(f)
        return JSONResponse(content=log_content)
    except Exception as e:
        print(f"MAIN_APP: Error reading {log_file}: {e}")
        return JSONResponse(content=[{"error": f"Failed to parse {log_file.name}"}], status_code=200)

@app.get("/agent/camera/status")
async def get_camera_agent_status():
    if camera_agent_instance and camera_agent_instance.running:
        return {"status": "running", "monitoring": camera_agent_instance.video_monitor.cap is not None and camera_agent_instance.video_monitor.cap.isOpened()}
    return {"status": "stopped"}

if __name__ == "__main__":
    # This allows running with `python -m streaming-test2.app.main` from repository root
    # or `python main.py` from `streaming-test2/app/`
    print("MAIN_APP: To run this app, use Uvicorn, e.g.:")
    print("uvicorn streaming-test2.app.main:app --reload --port 8000 --host 0.0.0.0 --app-dir . (from repo root)")
    # For direct execution `python streaming-test2/app/main.py` (from repo root):
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="streaming-test2/app")
