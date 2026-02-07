const API_BASE = `${window.location.protocol}//${window.location.host}/api`;
const WS_URL = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws/logs`;

// DOM Elements
const elStatus = document.getElementById("global-status");
const elUrl = document.getElementById("url-input");
const btnStart = document.getElementById("btn-start");
const btnContinue = document.getElementById("btn-continue");
const btnStop = document.getElementById("btn-stop");
const btnReset = document.getElementById("btn-reset");
const btnRetryBest = document.getElementById("btn-retry-best");
const btnBurn = document.getElementById("btn-burn");
const btnUpload = document.getElementById("btn-upload");
const fileInput = document.getElementById("file-upload");
const videoUploadInput = document.getElementById("video-upload");
const btnSelectVideo = document.getElementById("btn-select-video");
const selectedVideoName = document.getElementById("selected-video-name");
const consoleOutput = document.getElementById("console-output");
const fileList = document.getElementById("file-list");
const taskList = {
    dl360: document.getElementById("task-dl-360"),
    process: document.getElementById("task-process"),
    dlBest: document.getElementById("task-dl-best"),
    burn: document.getElementById("task-burn")
};

let ws;
let isTaskRunning = false;
const localInputState = {
    mode: "empty", // empty | uploading | ready | error
    fileName: "",
    progress: 0
};

const STATUS_MAP = {
    IDLE: "空闲",
    DOWNLOADING_360P: "下载(360p)",
    PROCESSING: "处理中",
    DOWNLOADING_BEST: "下载(最佳)",
    COMPLETED: "完成",
    ERROR: "错误"
};

const TASK_STATUS_MAP = {
    pending: "等待中",
    running: "运行中",
    completed: "已完成",
    error: "错误",
    stopped: "已停止"
};

const elConnection = document.createElement("span");
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

function logToConsole(text) {
    if (!text) return;
    const lines = text.split("\n");
    lines.forEach((line) => {
        if (!line) return;

        const lastChild = consoleOutput.lastElementChild;
        const isProgressLine = line.includes("%") || line.includes("it/s");
        const lastIsProgress = lastChild && (lastChild.textContent.includes("%") || lastChild.textContent.includes("it/s"));
        if (lastChild && isProgressLine && lastIsProgress) {
            lastChild.textContent = line;
            return;
        }

        const div = document.createElement("div");
        div.className = "log-line";
        div.textContent = line;
        consoleOutput.appendChild(div);
    });
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function connectWebSocket() {
    updateConnectionStatus("CLOSED");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        updateConnectionStatus("OPEN");
        logToConsole(">> 日志服务已连接");
    };

    ws.onmessage = (event) => {
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

async function apiCall(endpoint, method = "POST", body = null) {
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
    const contentType = res.headers.get("content-type") || "";
    let payload = null;

    if (contentType.includes("application/json")) {
        payload = await res.json();
    } else {
        const text = await res.text();
        if (text) payload = { message: text };
    }

    if (!res.ok) {
        throw new Error(payload?.message || `HTTP ${res.status}`);
    }
    return payload;
}

function syncButtonDisabledStates() {
    const busyUploading = localInputState.mode === "uploading";
    btnStart.disabled = isTaskRunning || busyUploading;
    btnSelectVideo.disabled = isTaskRunning || busyUploading;
}

function renderLocalInputButton() {
    btnSelectVideo.classList.remove("upload-progress");
    btnSelectVideo.style.removeProperty("--upload-progress");

    if (localInputState.mode === "uploading") {
        const p = Math.max(0, Math.min(100, localInputState.progress || 0));
        btnSelectVideo.classList.add("upload-progress");
        btnSelectVideo.style.setProperty("--upload-progress", `${p}%`);
        btnSelectVideo.textContent = `上传中 ${p}%`;
    } else if (localInputState.mode === "ready") {
        btnSelectVideo.textContent = "更换文件";
    } else if (localInputState.mode === "error") {
        btnSelectVideo.textContent = "重试上传";
    } else {
        btnSelectVideo.textContent = "选择文件";
    }

    if (localInputState.mode === "ready" && localInputState.fileName) {
        selectedVideoName.textContent = localInputState.fileName;
    } else if (localInputState.mode === "uploading" && localInputState.fileName) {
        selectedVideoName.textContent = localInputState.fileName;
    } else if (localInputState.mode === "error") {
        selectedVideoName.textContent = "上传失败";
    } else {
        selectedVideoName.textContent = "未选择文件";
    }

    syncButtonDisabledStates();
}

function setLocalInputState(mode, fileName = "", progress = 0) {
    localInputState.mode = mode;
    localInputState.fileName = fileName || "";
    localInputState.progress = progress || 0;
    renderLocalInputButton();
}

async function refreshLocalInputState() {
    try {
        const data = await apiCall("/local_input/state", "GET");
        if (data?.state === "ready" && data.file?.name) {
            setLocalInputState("ready", data.file.name, 100);
        } else {
            setLocalInputState("empty");
        }
    } catch (err) {
        setLocalInputState("empty");
        logToConsole(`!! 本地缓存状态异常: ${err.message}`);
    }
}

function uploadLocalInputWithProgress(file) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/local_input/upload`);

        xhr.upload.onprogress = (event) => {
            if (!event.lengthComputable) return;
            const percent = Math.min(99, Math.floor((event.loaded / event.total) * 100));
            setLocalInputState("uploading", file.name, percent);
        };

        xhr.onload = () => {
            const raw = xhr.responseText || "";
            let data = {};
            if (raw) {
                try {
                    data = JSON.parse(raw);
                } catch (_) {
                    data = {};
                }
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                setLocalInputState("uploading", file.name, 100);
                resolve(data);
                return;
            }

            reject(new Error(data.message || `HTTP ${xhr.status}`));
        };

        xhr.onerror = () => {
            reject(new Error("Network error"));
        };

        const formData = new FormData();
        formData.append("file", file);
        xhr.send(formData);
    });
}

async function updateStatus() {
    try {
        const data = await apiCall("/status", "GET");
        if (!data) return;

        elStatus.textContent = STATUS_MAP[data.status] || data.status;
        updateTaskItem(taskList.dl360, data.tasks.download_360p);
        updateTaskItem(taskList.process, data.tasks.process_transcription);
        updateTaskItem(taskList.dlBest, data.tasks.download_best);
        updateTaskItem(taskList.burn, data.tasks.burn_video);

        isTaskRunning = data.status !== "IDLE" && data.status !== "COMPLETED" && data.status !== "ERROR";
        syncButtonDisabledStates();
    } catch (err) {
        console.error(err);
    }
}

function updateTaskItem(el, taskInfo) {
    if (!el || !taskInfo) return;
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

async function updateFiles() {
    try {
        const files = await apiCall("/files", "GET");
        if (!files) return;

        fileList.innerHTML = "";
        files.forEach((f) => {
            const row = document.createElement("div");
            row.className = "file-item";

            const name = document.createElement("span");
            name.textContent = f.name;

            const action = document.createElement("span");
            const link = document.createElement("a");
            link.href = `${API_BASE}/download/${encodeURIComponent(f.name)}`;
            link.target = "_blank";
            link.textContent = "[下载]";
            action.appendChild(link);

            row.appendChild(name);
            row.appendChild(action);
            fileList.appendChild(row);
        });
    } catch (err) {
        console.error(err);
    }
}

document.getElementById("btn-clear-log").addEventListener("click", () => {
    consoleOutput.innerHTML = "";
    logToConsole(">> 屏幕已清空");
});

btnSelectVideo.addEventListener("click", () => {
    if (btnSelectVideo.disabled) return;
    videoUploadInput.click();
});

videoUploadInput.addEventListener("change", async () => {
    if (!videoUploadInput.files || videoUploadInput.files.length === 0) return;

    const file = videoUploadInput.files[0];
    elUrl.value = "";
    setLocalInputState("uploading", file.name, 0);
    logToConsole(`>> 开始上传本地文件: ${file.name}`);

    try {
        const result = await uploadLocalInputWithProgress(file);
        const savedName = result?.filename || file.name;
        setLocalInputState("ready", savedName, 100);
        logToConsole(`>> 本地文件已缓存: ${savedName}`);
    } catch (err) {
        setLocalInputState("error", file.name, 0);
        logToConsole(`!! 本地文件上传失败: ${err.message}`);
    } finally {
        videoUploadInput.value = "";
        await refreshLocalInputState();
    }
});

btnStart.addEventListener("click", async () => {
    const url = elUrl.value.trim();

    if (localInputState.mode === "uploading") {
        logToConsole("!! 文件上传中，请稍后再开始任务");
        return;
    }

    try {
        if (url) {
            await apiCall("/start", "POST", { url });
            logToConsole(">> 已按链接模式开始任务");
            return;
        }

        await apiCall("/start_local", "POST");
        logToConsole(">> 已按本地缓存模式开始任务");
    } catch (err) {
        logToConsole(`!! 启动失败: ${err.message}`);
    }
});

btnContinue.addEventListener("click", async () => {
    if (!confirm("继续上次任务？将从中断步骤继续执行。")) return;
    try {
        await apiCall("/continue", "POST");
    } catch (err) {
        logToConsole(`!! 继续任务失败: ${err.message}`);
    }
});

btnStop.addEventListener("click", async () => {
    if (!confirm("警告：确定要强制停止当前任务吗？")) return;
    try {
        await apiCall("/stop", "POST");
    } catch (err) {
        logToConsole(`!! 停止任务失败: ${err.message}`);
    }
});

btnReset.addEventListener("click", async () => {
    if (!confirm("警告：确定要归档当前任务并重置吗？")) return;
    try {
        await apiCall("/reset", "POST");
    } catch (err) {
        logToConsole(`!! 重置失败: ${err.message}`);
    }
});

if (btnRetryBest) {
    btnRetryBest.addEventListener("click", async () => {
        const url = elUrl.value.trim();
        const payload = url ? { url } : {};

        try {
            const resp = await apiCall("/retry_best", "POST", payload);
            logToConsole(`>> ${resp?.message || "最佳视频重试已启动"}`);
        } catch (err) {
            logToConsole(`!! 重试最佳视频下载失败: ${err.message}`);
        }
    });
}

btnBurn.addEventListener("click", async () => {
    try {
        await apiCall("/burn", "POST");
    } catch (err) {
        logToConsole(`!! 烧录失败: ${err.message}`);
    }
});

btnUpload.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        logToConsole(`>> 正在上传字幕: ${file.name}`);
        await apiCall("/upload_sub", "POST", formData);
        logToConsole(">> 字幕上传完成");
    } catch (err) {
        logToConsole(`!! 字幕上传失败: ${err.message}`);
    }
});

connectWebSocket();
setLocalInputState("empty");
updateStatus();
updateFiles();
refreshLocalInputState();

setInterval(updateStatus, 1000);
setInterval(updateFiles, 5000);
setInterval(refreshLocalInputState, 5000);
