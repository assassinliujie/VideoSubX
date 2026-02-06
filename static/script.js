const API_BASE = "http://" + window.location.host + "/api";
const WS_URL = "ws://" + window.location.host + "/ws/logs";

// DOM Elements
const elStatus = document.getElementById("global-status");
const elUrl = document.getElementById("url-input");
const btnStart = document.getElementById("btn-start");
const btnContinue = document.getElementById("btn-continue");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");
const btnBurn = document.getElementById("btn-burn");
const btnUpload = document.getElementById("btn-upload");
const fileInput = document.getElementById("file-upload");
const videoUploadInput = document.getElementById("video-upload");
const btnSelectVideo = document.getElementById("btn-select-video");
const selectedVideoName = document.getElementById("selected-video-name");
const consoleOutput = document.getElementById("console-output");
const taskList = {
    dl360: document.getElementById("task-dl-360"),
    process: document.getElementById("task-process"),
    dlBest: document.getElementById("task-dl-best"),
    burn: document.getElementById("task-burn")
};
const fileList = document.getElementById("file-list");

// 本地文件上传状态
let uploadedVideoFile = null;

// WebSocket for Logs
let ws;
const elConnection = document.createElement("span");
// Add connection indicator to header dynamically
elConnection.style.fontSize = "0.5em";
elConnection.style.marginLeft = "10px";
document.querySelector(".header h1").appendChild(elConnection);

function updateConnectionStatus(status) {
    if (status === "OPEN") {
        elConnection.textContent = "● ONLINE";
        elConnection.style.color = "#55ff55";
        elConnection.style.textShadow = "0 0 5px #55ff55";
    } else {
        elConnection.textContent = "● OFFLINE";
        elConnection.style.color = "#ff3333";
        elConnection.style.textShadow = "0 0 5px #ff3333";
    }
}

function connectWebSocket() {
    updateConnectionStatus("CLOSED");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        logToConsole(">> 连接建立: 日志服务器已连接");
        updateConnectionStatus("OPEN");
    };

    ws.onmessage = (event) => {
        // Handle incoming log block
        logToConsole(event.data);
    };

    ws.onclose = () => {
        updateConnectionStatus("CLOSED");
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
        console.error("WebSocket Error:", err);
        updateConnectionStatus("CLOSED");
    };
}

function logToConsole(text) {
    if (!text) return;

    const lines = text.split('\n');
    lines.forEach(line => {
        if (!line) return;

        // Progress Bar Update Logic
        const lastChild = consoleOutput.lastElementChild;
        // Check if both lines look like progress updates to just replace text
        if (lastChild &&
            (line.includes('%') || line.includes('it/s')) &&
            (lastChild.textContent.includes('%') || lastChild.textContent.includes('it/s'))) {
            lastChild.textContent = line;
        } else {
            // Standard append
            const div = document.createElement("div");
            div.className = "log-line";
            div.textContent = line;
            consoleOutput.appendChild(div);
        }
    });

    // Force Scroll
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

document.getElementById("btn-clear-log").addEventListener("click", () => {
    consoleOutput.innerHTML = "";
    logToConsole(">> 屏幕已清空");
});

// API Calls
async function apiCall(endpoint, method = "POST", body = null) {
    try {
        const options = {
            method,
            headers: {}
        };
        if (body) {
            if (body instanceof FormData) {
                options.body = body;
            } else {
                options.headers["Content-Type"] = "application/json";
                options.body = JSON.stringify(body);
            }
        }
        const res = await fetch(API_BASE + endpoint, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        logToConsole(`!! API错误: ${endpoint} ${err.message}`);
        // Let caller handle specific alerts if needed
        throw err;
    }
}

// Polling Status
const STATUS_MAP = {
    "IDLE": "空闲",
    "DOWNLOADING_360P": "下载(360p)",
    "PROCESSING": "处理中",
    "DOWNLOADING_BEST": "下载(最佳)",
    "COMPLETED": "完成",
    "ERROR": "错误"
};

const TASK_STATUS_MAP = {
    "pending": "等待中",
    "running": "运行中",
    "completed": "已完成",
    "error": "错误",
    "stopped": "已停止"
};

async function updateStatus() {
    try {
        const data = await apiCall("/status", "GET");
        if (!data) return;

        elStatus.textContent = STATUS_MAP[data.status] || data.status;

        // Update Task List
        updateTaskItem(taskList.dl360, data.tasks.download_360p);
        updateTaskItem(taskList.process, data.tasks.process_transcription);
        updateTaskItem(taskList.dlBest, data.tasks.download_best);
        updateTaskItem(taskList.burn, data.tasks.burn_video);

        // Update Buttons
        const running = data.status !== "IDLE" && data.status !== "COMPLETED" && data.status !== "ERROR";
        btnStart.disabled = running;

    } catch (e) {
        console.error(e);
    }
}

function updateTaskItem(el, taskInfo) {
    if (!taskInfo) return;
    const checkbox = el.querySelector(".checkbox");
    const statusText = el.querySelector(".status");

    statusText.textContent = TASK_STATUS_MAP[taskInfo.status] || taskInfo.status;
    statusText.className = "status " + taskInfo.status;

    if (taskInfo.status === "completed") {
        checkbox.textContent = "[X]";
        el.style.opacity = "0.5";
    } else if (taskInfo.status === "running") {
        checkbox.textContent = "[>]";
        el.style.opacity = "1";
    } else if (taskInfo.status === "error") {
        checkbox.textContent = "[!]";
        el.style.opacity = "1";
    } else {
        checkbox.textContent = "[ ]";
        el.style.opacity = "1";
    }
}

// File List
async function updateFiles() {
    const files = await apiCall("/files", "GET");
    if (!files) return;

    fileList.innerHTML = "";
    files.forEach(f => {
        const div = document.createElement("div");
        div.className = "file-item";
        div.innerHTML = `
            <span>${f.name}</span>
            <span><a href="${API_BASE}/download/${f.name}" target="_blank">[下载]</a></span>
        `;
        fileList.appendChild(div);
    });
}

// Event Listeners

// 选择本地视频文件
btnSelectVideo.addEventListener("click", () => {
    videoUploadInput.click();
});

videoUploadInput.addEventListener("change", () => {
    if (videoUploadInput.files.length > 0) {
        uploadedVideoFile = videoUploadInput.files[0];
        selectedVideoName.textContent = uploadedVideoFile.name;
        // 清空 URL 输入框，表示使用本地文件模式
        elUrl.value = "";
        logToConsole(`>> 已选择本地文件: ${uploadedVideoFile.name}`);
    } else {
        uploadedVideoFile = null;
        selectedVideoName.textContent = "未选择文件";
    }
});

// URL 输入时清空已选择的本地文件
elUrl.addEventListener("input", () => {
    if (elUrl.value.trim()) {
        uploadedVideoFile = null;
        videoUploadInput.value = "";
        selectedVideoName.textContent = "未选择文件";
    }
});

btnStart.addEventListener("click", async () => {
    const url = elUrl.value.trim();

    // 检查是否有本地文件或 URL
    if (!url && !uploadedVideoFile) {
        logToConsole("!! 错误: 请输入视频链接或选择本地文件");
        return;
    }

    if (uploadedVideoFile) {
        // 本地文件模式：先上传文件，再启动本地处理
        logToConsole(`>> 正在上传本地文件: ${uploadedVideoFile.name}...`);

        const formData = new FormData();
        formData.append("file", uploadedVideoFile);

        try {
            const uploadResult = await apiCall("/upload_video", "POST", formData);
            logToConsole(`>> 文件上传成功: ${uploadResult.filename}`);

            // 启动本地处理
            await apiCall("/start_local", "POST");

            // 清空已选择的文件
            uploadedVideoFile = null;
            videoUploadInput.value = "";
            selectedVideoName.textContent = "未选择文件";
        } catch (err) {
            logToConsole(`!! 上传或启动失败: ${err.message}`);
        }
    } else {
        // URL 模式：使用原有逻辑
        await apiCall("/start", "POST", { url });
    }
});

btnContinue.addEventListener("click", async () => {
    // 继续任务不需要 URL，直接调用 continue API
    if (confirm("继续上次任务？将从上次中断的步骤继续执行。")) {
        await apiCall("/continue", "POST");
    }
});

btnStop.addEventListener("click", async () => {
    if (confirm("警告: 确定要强制停止所有任务吗？")) {
        await apiCall("/stop", "POST");
    }
});

btnReset.addEventListener("click", async () => {
    if (confirm("警告: 确定要归档当前任务并重置吗？")) {
        await apiCall("/reset", "POST");
    }
});

btnBurn.addEventListener("click", async () => {
    await apiCall("/burn", "POST");
});

btnUpload.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    logToConsole(`>> 正在上传: ${file.name}...`);
    await apiCall("/upload_sub", "POST", formData);
    logToConsole(">> 上传完成");
});

// Initialization
connectWebSocket();
setInterval(updateStatus, 1000);
setInterval(updateFiles, 5000);
updateFiles();
