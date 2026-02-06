import asyncio
from enum import Enum
from typing import List, Dict, Any
from collections import deque
from datetime import datetime

class TaskStatus(str, Enum):
    IDLE = "IDLE"
    DOWNLOADING_360P = "DOWNLOADING_360P"
    PROCESSING = "PROCESSING"
    DOWNLOADING_BEST = "DOWNLOADING_BEST"  # Can run in parallel with PROCESSING
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

class GlobalState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalState, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.status = TaskStatus.IDLE
        self.logs = deque(maxlen=2000)  # Keep last 2000 lines
        self.log_count = 0 # Monotonic counter for reliable polling
        self.tasks = {
            "download_360p": {"status": "pending", "progress": 0},
            "process_transcription": {"status": "pending", "progress": 0},
            "download_best": {"status": "pending", "progress": 0},
            "burn_video": {"status": "pending", "progress": 0}
        }
        self.error_msg = None
        self.subscribers = []

    def update_task_status(self, task_name: str, status: str):
        if task_name in self.tasks:
            self.tasks[task_name]["status"] = status
            self.notify_subscribers()

    def set_status(self, status: TaskStatus):
        self.status = status
        self.notify_subscribers()

    def add_log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        self.log_count += 1
        # Notify logging listeners (handled via WebSocket usually)

    def reset(self):
        self.status = TaskStatus.IDLE
        self.tasks = {
            "download_360p": {"status": "pending", "progress": 0},
            "process_transcription": {"status": "pending", "progress": 0},
            "download_best": {"status": "pending", "progress": 0},
            "burn_video": {"status": "pending", "progress": 0}
        }
        self.error_msg = None
        self.add_log("System reset.")
        self.notify_subscribers()
        
    def notify_subscribers(self):
        # This will be used to push updates to WebSockets
        pass

# Singleton instance
state = GlobalState()
