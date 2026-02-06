import os
import sys
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Ensure core is in path structure
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from backend.global_state import state, TaskStatus
from backend.task_manager import task_manager
from backend.logger import setup_logger

# Initialize Logger to capture stdout/stderr
setup_logger()

app = FastAPI(title="VideoSubX WebUI", description="FastAPI Backend for VideoSubX", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class StartRequest(BaseModel):
    url: str

# API Endpoints
@app.post("/api/start")
async def start_task(req: StartRequest):
    if state.status != TaskStatus.IDLE and state.status != TaskStatus.COMPLETED and state.status != TaskStatus.ERROR:
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
        "error": state.error_msg
    }

@app.post("/api/reset")
async def reset_task():
    task_manager.reset_workspace()
    return {"message": "Workspace reset"}

@app.post("/api/continue")
async def continue_task():
    """继续上次中断的任务，从断点处继续执行"""
    if state.status != TaskStatus.IDLE and state.status != TaskStatus.COMPLETED and state.status != TaskStatus.ERROR:
        return JSONResponse(status_code=400, content={"message": "Task already running"})
    
    task_manager.continue_workflow()
    return {"message": "Continue task started", "status": state.status}

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
    """上传本地视频/音频文件"""
    from core.utils import load_key
    
    # 获取允许的视频格式
    allowed_formats = load_key("allowed_video_formats")
    if not allowed_formats:
        allowed_formats = ["mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "mp3", "wav", "m4a", "flac"]
    
    # 检查文件扩展名
    ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if ext not in allowed_formats:
        return JSONResponse(status_code=400, content={
            "message": f"不支持的文件格式: .{ext}。支持的格式: {', '.join(allowed_formats)}"
        })
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理文件名并添加 _best 后缀
    import re
    base_name = os.path.splitext(file.filename)[0]
    # 移除或替换非法字符
    base_name = re.sub(r'[<>:"/\\|?*]', '', base_name).strip('. ')
    if not base_name:
        base_name = 'video'
    
    # 保存文件，带 _best 后缀
    save_filename = f"{base_name}_best.{ext}"
    file_path = os.path.join(output_dir, save_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    state.add_log(f"User uploaded video file: {file.filename} -> {save_filename}")
    return {"message": "Video uploaded successfully", "filename": save_filename}

@app.post("/api/start_local")
async def start_local():
    """使用上传的本地文件开始处理（跳过下载步骤）"""
    if state.status != TaskStatus.IDLE and state.status != TaskStatus.COMPLETED and state.status != TaskStatus.ERROR:
        return JSONResponse(status_code=400, content={"message": "Task already running"})
    
    # 检查 output 目录是否有视频文件
    from core.downloader import find_video_files
    try:
        video_file = find_video_files()
        state.add_log(f"Found local video file: {video_file}")
    except ValueError:
        return JSONResponse(status_code=400, content={"message": "No video file found in output directory. Please upload a video first."})
    
    task_manager.start_local_workflow()
    return {"message": "Local processing started", "status": state.status}

@app.get("/api/files")
async def list_files():
    output_dir = "output"
    if not os.path.exists(output_dir):
        return []
    
    files = []
    for f in os.listdir(output_dir):
        path = os.path.join(output_dir, f)
        if os.path.isfile(path):
            files.append({
                "name": f,
                "size": os.path.getsize(path),
                "time": os.path.getmtime(path)
            })
    return files

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return JSONResponse(status_code=404, content={"message": "File not found"})

# WebSocket for Logs


# Improved WebSocket logic needs GlobalState support
# Let's patching GlobalState to support async queue subscription
# But we can't patch it here easily.
# So I'll modify the WebSocket logic to just push the *entire* log whenever it changes?
# No, that's bad.
# 
# Let's make a simple LogBroadcaster.
class LogBroadcaster:
    def __init__(self):
        self.connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        # Send history
        try:
             # Convert deque to list for JSON serialization or just strings
             history = "\n".join(state.logs)
             await websocket.send_text(history)
        except:
             pass

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except:
                pass

broadcaster = LogBroadcaster()

# Monkey patch GlobalState to notify broadcaster? 
# Or just have a background task in FastAPI that watches for logs.
# Let's do a background task.

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(log_watcher())

async def log_watcher():
    last_count = 0
    while True:
        current_len = len(state.logs)
        if current_len > last_count:
             # Identify new lines. 
             # If deque rotated, this logic is tricky. 
             # Let's assuming <2000 lines for this session or just send the new line if we catch it.
             # Actually, simpler:
             # In `logger.py` or `global_state.py`, we call an async callback? 
             # No, can't call async from sync easily.
             
             # Let's just check the last item.
             # If it's different from what we last sent, send it?
             # But multiple lines could come in at once.
             
             # Compromise:
             # We will just send the whole log buffer every 1 second if it changed?
             # No, too heavy.
             
             # Let's try to get a snapshot.
             pass
             
        # Re-evaluating:
        # The prompt asked for "WebSocket" to receive logs.
        # I will implement a smarter poll.
        # State will carry a 'log_version' counter.
        pass
        
        await asyncio.sleep(0.1)

# Redefining the endpoint to use a simpler queue approach
# I'll just rely on the client refreshing full logs on connect, 
# and then we just push updates.

# New approach: 
# We'll use a globally available async queue? 
# No, `state.add_log` is sync.
# 
# Final Decision: I will poll `state.logs` intelligently.
# I will keep track of the *id* of the last log sent.
# I'll change GlobalState to store logs as (id, message).

# But I can't easily change GlobalState now without rewriting the file.
# Wait, I CAN rewrite the file or just use the deque interactions.
# 
# Let's accept that for this MVP, the WebSocket might functionality is:
# 1. On connect, send all logs.
# 2. Every 0.5s, check if len(state.logs) > last_len. 
#    - If so, send the new items (by slicing list(state.logs)[last_len:]).
#    - Update last_len.
#    - If len < last_len (rotation), reset last_len = 0 and send all.

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    
    # Send all history initially
    history = list(state.logs)
    if history:
        await websocket.send_text("\n".join(history))
    
    # Start tracking from the global counter corresponding to the end of current history
    # The safest way is to just use state.log_count directly
    last_count = state.log_count
    
    try:
        while True:
            await asyncio.sleep(0.1) # 100ms polling
            
            current_count = state.log_count
            
            if current_count > last_count:
                # Calculate how many new logs we have
                diff = current_count - last_count
                
                # Get the last 'diff' logs from the deque
                # Use list slicing on the deque (copying is acceptable for small batches)
                # If diff is huge (larger than deque size), we might have missed some
                # but we can only send what's in the deque.
                
                logs_snapshot = list(state.logs)
                if diff > len(logs_snapshot):
                    new_logs = logs_snapshot # Send everything we have
                else:
                    new_logs = logs_snapshot[-diff:]
                
                if new_logs:
                    await websocket.send_text("\n".join(new_logs))
                
                last_count = current_count
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")

# Static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Listen on 0.0.0.0 because users might access from LAN or just localhost
    uvicorn.run(app, host="0.0.0.0", port=8501)
