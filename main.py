import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from backend.global_state import state, TaskStatus
from backend.logger import setup_logger
from backend.task_manager import task_manager

# Initialize logger to capture stdout/stderr
setup_logger()

app = FastAPI(
    title="VideoSubX WebUI",
    description="FastAPI Backend for VideoSubX",
    version="2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    url: str


class RetryBestRequest(BaseModel):
    url: Optional[str] = None


LOCAL_INPUT_DIR = Path("runtime/local_input")


def _is_task_running() -> bool:
    return state.status not in {TaskStatus.IDLE, TaskStatus.COMPLETED, TaskStatus.ERROR}


def _load_allowed_formats():
    from core.utils import load_key

    allowed_formats = load_key("allowed_video_formats")
    if not allowed_formats:
        allowed_formats = ["mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "mp3", "wav", "m4a", "flac"]
    return [ext.lower().lstrip(".") for ext in allowed_formats]


def _sanitize_base_name(filename: str) -> str:
    import re

    base_name = os.path.splitext(os.path.basename(filename))[0]
    base_name = re.sub(r'[<>:"/\\|?*]', "", base_name).strip(". ")
    return base_name or "video"


def _ensure_local_input_dir():
    LOCAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def _list_local_input_video_files():
    _ensure_local_input_dir()
    allowed_set = set(_load_allowed_formats())
    files = []
    for path in LOCAL_INPUT_DIR.iterdir():
        if path.is_file() and path.suffix.lower().lstrip(".") in allowed_set:
            files.append(path)
    return sorted(files, key=lambda p: p.name.lower())


def _clear_local_input_dir() -> int:
    _ensure_local_input_dir()
    removed = 0
    for path in LOCAL_INPUT_DIR.iterdir():
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
            elif path.is_dir():
                shutil.rmtree(path)
                removed += 1
        except Exception:
            continue
    return removed


def _resolve_single_local_input_file(clear_if_multiple: bool = False):
    files = _list_local_input_video_files()
    if len(files) == 1:
        return files[0], 1, False
    if len(files) == 0:
        return None, 0, False

    if clear_if_multiple:
        _clear_local_input_dir()
        task_manager.clear_local_video()
        return None, len(files), True
    return None, len(files), False


@app.post("/api/start")
async def start_task(req: StartRequest):
    if _is_task_running():
        return JSONResponse(status_code=400, content={"message": "Task already running"})

    task_manager.start_workflow(req.url)
    return {"message": "Task started", "status": state.status}


@app.post("/api/stop")
async def stop_task():
    task_manager.stop_workflow()
    return {"message": "Stop signal sent"}


@app.get("/api/status")
async def get_status():
    return {
        "status": state.status,
        "tasks": state.tasks,
        "error": state.error_msg,
    }


@app.post("/api/reset")
async def reset_task():
    task_manager.reset_workspace()
    return {"message": "Workspace reset"}


@app.post("/api/continue")
async def continue_task():
    if _is_task_running():
        return JSONResponse(status_code=400, content={"message": "Task already running"})

    task_manager.continue_workflow()
    return {"message": "Continue task started", "status": state.status}


@app.post("/api/retry_best")
async def retry_best_task(req: Optional[RetryBestRequest] = None):
    if _is_task_running():
        return JSONResponse(status_code=400, content={"message": "Task already running"})

    retry_url = ((req.url if req else "") or "").strip() or None
    ok, message = task_manager.retry_download_best(retry_url)
    if not ok:
        return JSONResponse(status_code=400, content={"message": message})

    return {"message": message, "status": state.status}


@app.post("/api/burn")
async def burn_video():
    task_manager.burn_video()
    return {"message": "Burn task started"}


@app.post("/api/upload_sub")
async def upload_sub(file: UploadFile = File(...)):
    if not file.filename.endswith(".ass"):
        return JSONResponse(status_code=400, content={"message": "Only .ass files allowed"})

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "src_trans.ass")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    state.add_log(f"User uploaded new subtitle file: {file.filename}")
    return {"message": "Subtitle uploaded successfully"}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    allowed_formats = _load_allowed_formats()

    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in allowed_formats:
        return JSONResponse(
            status_code=400,
            content={"message": f"Unsupported file format: .{ext}. Allowed: {', '.join(allowed_formats)}"},
        )

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    base_name = _sanitize_base_name(file.filename)
    save_filename = f"{base_name}_best.{ext}"
    file_path = os.path.join(output_dir, save_filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Remember the uploaded local file so local-start mode preserves exactly this one.
    task_manager.set_local_video(save_filename)

    state.add_log(f"User uploaded video file: {file.filename} -> {save_filename}")
    return {"message": "Video uploaded successfully", "filename": save_filename}


@app.get("/api/local_input/state")
async def get_local_input_state():
    local_file, file_count, cleared = _resolve_single_local_input_file(clear_if_multiple=True)

    if file_count > 1:
        state.add_log("Detected multiple files in local input cache. Cleared cache for safety.")
        return JSONResponse(
            status_code=400,
            content={
                "state": "invalid_multiple",
                "has_file": False,
                "message": "Multiple files detected in local cache; cache was cleared. Please upload again.",
                "cleared": cleared,
            },
        )

    if not local_file:
        task_manager.clear_local_video()
        return {"state": "empty", "has_file": False, "file": None}

    stat = local_file.stat()
    return {
        "state": "ready",
        "has_file": True,
        "file": {
            "name": local_file.name,
            "size": stat.st_size,
            "time": stat.st_mtime,
        },
    }


@app.delete("/api/local_input")
async def clear_local_input():
    removed = _clear_local_input_dir()
    task_manager.clear_local_video()
    state.add_log("Local input cache cleared.")
    return {"message": "Local input cache cleared", "removed": removed}


@app.post("/api/local_input/upload")
async def upload_local_input(file: UploadFile = File(...)):
    if _is_task_running():
        return JSONResponse(status_code=400, content={"message": "Task already running"})

    allowed_formats = _load_allowed_formats()
    ext = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if ext not in allowed_formats:
        return JSONResponse(
            status_code=400,
            content={"message": f"Unsupported file format: .{ext}. Allowed: {', '.join(allowed_formats)}"},
        )

    _clear_local_input_dir()

    base_name = _sanitize_base_name(file.filename)
    save_filename = f"{base_name}_best.{ext}"
    save_path = LOCAL_INPUT_DIR / save_filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task_manager.set_local_video(save_filename, source_path=str(save_path))
    state.add_log(f"Cached local input file: {file.filename} -> {save_filename}")
    return {"message": "Local input uploaded successfully", "filename": save_filename}


@app.post("/api/start_local")
async def start_local():
    if _is_task_running():
        return JSONResponse(status_code=400, content={"message": "Task already running"})

    local_file, file_count, cleared = _resolve_single_local_input_file(clear_if_multiple=True)
    if file_count > 1:
        state.add_log("Multiple files found in local cache. Cache cleared; waiting for re-upload.")
        return JSONResponse(
            status_code=400,
            content={
                "message": "Multiple files found in local cache. Cache was cleared, please upload one file again.",
                "cleared": cleared,
            },
        )

    if not local_file:
        return JSONResponse(
            status_code=400,
            content={"message": "No cached local file found. Please upload a local file first."},
        )

    state.add_log(f"Using cached local file: {local_file.name}")
    task_manager.start_local_workflow(
        local_video_filename=local_file.name,
        local_video_source_path=str(local_file),
    )
    return {"message": "Local processing started", "status": state.status}


@app.get("/api/files")
async def list_files():
    output_dir = "output"
    if not os.path.exists(output_dir):
        return []

    files = []
    for filename in os.listdir(output_dir):
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            files.append({
                "name": filename,
                "size": os.path.getsize(path),
                "time": os.path.getmtime(path),
            })
    return files


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        return JSONResponse(status_code=400, content={"message": "Invalid filename"})

    output_dir = Path("output").resolve()
    file_path = (output_dir / safe_name).resolve()

    if not str(file_path).startswith(str(output_dir) + os.sep):
        return JSONResponse(status_code=400, content={"message": "Invalid filename"})

    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path), filename=safe_name)

    return JSONResponse(status_code=404, content={"message": "File not found"})


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()

    history = list(state.logs)
    if history:
        await websocket.send_text("\n".join(history))

    last_count = state.log_count

    try:
        while True:
            await asyncio.sleep(0.1)

            current_count = state.log_count
            if current_count > last_count:
                diff = current_count - last_count
                logs_snapshot = list(state.logs)
                if diff > len(logs_snapshot):
                    new_logs = logs_snapshot
                else:
                    new_logs = logs_snapshot[-diff:]

                if new_logs:
                    await websocket.send_text("\n".join(new_logs))

                last_count = current_count

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")


app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
