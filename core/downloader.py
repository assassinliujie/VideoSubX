import os,sys
import glob
import re
import subprocess
import time
from core.utils import *

def sanitize_filename(filename):
    # Remove or replace illegal characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Ensure filename doesn't start or end with a dot or space
    filename = filename.strip('. ')
    # Use default name if filename is empty
    return filename if filename else 'video'

def update_ytdlp():
    """
    智能更新 yt-dlp：
    1. 如果配置了代理，使用代理更新
    2. 如果没有代理，检测官方 PyPI 是否可访问
    3. 如果不可访问，使用清华镜像更新
    """
    import urllib.request
    import socket
    
    proxy = load_key("proxy")
    
    # 构建 pip 命令的基础参数
    pip_args = [sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"]
    
    if proxy:
        # 有代理配置，使用代理
        rprint(f"[blue]使用代理更新 yt-dlp: {proxy}[/blue]")
        pip_args.extend(["--proxy", proxy])
    else:
        # 无代理，检测网络
        def can_reach_pypi():
            """检测是否能访问官方 PyPI"""
            try:
                req = urllib.request.Request(
                    "https://pypi.org/simple/yt-dlp/",
                    headers={"User-Agent": "pip/23.0"}
                )
                urllib.request.urlopen(req, timeout=5)
                return True
            except (urllib.error.URLError, socket.timeout):
                return False
        
        if can_reach_pypi():
            rprint("[blue]检测到可访问 PyPI，使用官方源更新 yt-dlp...[/blue]")
        else:
            # 使用清华镜像
            rprint("[yellow]无法访问 PyPI，使用清华镜像更新 yt-dlp...[/yellow]")
            pip_args.extend(["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    
    try:
        subprocess.check_call(pip_args)
        if 'yt_dlp' in sys.modules:
            del sys.modules['yt_dlp']
        rprint("[green]yt-dlp 更新成功[/green]")
    except subprocess.CalledProcessError as e:
        rprint(f"[yellow]警告: yt-dlp 更新失败: {e}[/yellow]")
    
    from yt_dlp import YoutubeDL
    return YoutubeDL

def download_video_ytdlp(url, save_path='output', resolution='1080', suffix='', max_retries=2, retry_delay=2):
    os.makedirs(save_path, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best' if resolution == 'best' else 'worstvideo+bestaudio/best',
        'outtmpl': f'{save_path}/%(title)s{suffix}.%(ext)s',
        'noplaylist': True,
        'writethumbnail': True,
        'postprocessors': [{'key': 'FFmpegThumbnailsConvertor', 'format': 'jpg'}],
        'remote_components': ['ejs:github'],
    }
    
    # 代理设置：从配置文件读取，留空则不使用代理
    proxy = load_key("proxy")
    if proxy:
        ydl_opts['proxy'] = proxy

    # Read Youtube Cookie File
    cookies_path = load_key("youtube.cookies_path")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = str(cookies_path)

    # Get YoutubeDL class after updating
    YoutubeDL = update_ytdlp()
    total_attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, total_attempts + 1):
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            break
        except Exception as e:
            if attempt >= total_attempts:
                raise
            wait_seconds = max(0, int(retry_delay)) * attempt
            rprint(f"[yellow]Download failed (attempt {attempt}/{total_attempts}): {e}[/yellow]")
            rprint(f"[blue]Retrying download in {wait_seconds}s...[/blue]")
            if wait_seconds > 0:
                time.sleep(wait_seconds)
    
    # Check and rename files after download
    for file in os.listdir(save_path):
        if os.path.isfile(os.path.join(save_path, file)):
            filename, ext = os.path.splitext(file)
            new_filename = sanitize_filename(filename)
            if new_filename != filename:
                os.rename(os.path.join(save_path, file), os.path.join(save_path, new_filename + ext))

def download_video_async(url, save_path='output', resolution='1080', suffix=''):
    """异步下载视频，用于并行处理"""
    import threading
    def download_thread():
        try:
            download_video_ytdlp(url, save_path, resolution, suffix)
        except Exception as e:
            print(f"Download failed for {resolution}p: {e}")
    
    thread = threading.Thread(target=download_thread)
    thread.start()
    return thread

def find_video_files(save_path='output', prefer_best=True):
    video_files = [file for file in glob.glob(save_path + "/*") if os.path.splitext(file)[1][1:].lower() in load_key("allowed_video_formats")]
    # change \\ to /, this happen on windows
    if sys.platform.startswith('win'):
        video_files = [file.replace("\\", "/") for file in video_files]
    video_files = [file for file in video_files if not file.startswith("output/output")]
    
    if len(video_files) == 0:
        raise ValueError("No video files found in the output directory.")
    
    if len(video_files) == 1:
        return video_files[0]
    
    # If multiple videos found, prioritize best quality
    if prefer_best:
        best_files = [f for f in video_files if '_best.' in f or 'best.' in f]
        if best_files:
            return best_files[0]
    
    # Otherwise return the first one
    return video_files[0]

if __name__ == '__main__':
    # Example usage
    url = input('Please enter the URL of the video you want to download: ')
    resolution = input('Please enter the desired resolution (360/480/720/1080, default 1080): ')
    resolution = int(resolution) if resolution.isdigit() else 1080
    download_video_ytdlp(url, resolution=resolution)
    print(f"🎥 Video has been downloaded to {find_video_files()}")
