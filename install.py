"""
VideoSubX 一键安装脚本
自动检测系统环境并安装依赖
"""

import os
import sys
import platform
import subprocess

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ASCII_LOGO = """
__      __ _      _              _____  _    _  ____  __   __
\ \    / /(_)    | |            / ____|| |  | ||  _ \ \ \ / /
 \ \  / /  _   __| |  ___   ___| (___  | |  | || |_) | \ V /
  \ \/ /  | | / _` | / _ \ / _ \\___ \ | |  | ||  _ <   > <
   \  /   | || (_| ||  __/| (_) |___) || |__| || |_) | / . \
    \/    |_| \__,_| \___| \___/|_____/ \____/ |____/ /_/ \_\
	
"""

def run_pip(*args):
    """运行 pip 命令"""
    subprocess.check_call([sys.executable, "-m", "pip", *args])

def install_package(*packages):
    """安装 Python 包"""
    run_pip("install", *packages)

def check_nvidia_gpu():
    """检测 NVIDIA GPU"""
    try:
        install_package("pynvml")
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            print("✅ 检测到 NVIDIA GPU:")
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"   GPU {i}: {name}")
            pynvml.nvmlShutdown()
            return True
        pynvml.nvmlShutdown()
    except Exception:
        pass
    print("⚠️ 未检测到 NVIDIA GPU，将安装 CPU 版本")
    return False

def check_ffmpeg():
    """检查 FFmpeg 是否安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        print("✅ FFmpeg 已安装")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        system = platform.system()
        print("❌ 未找到 FFmpeg")
        print()
        if system == "Windows":
            print("   安装方式: choco install ffmpeg")
            print("   需要先安装 Chocolatey: https://chocolatey.org/")
        elif system == "Darwin":
            print("   安装方式: brew install ffmpeg")
            print("   需要先安装 Homebrew: https://brew.sh/")
        else:
            print("   安装方式: sudo apt install ffmpeg (Ubuntu/Debian)")
            print("             sudo yum install ffmpeg (CentOS/RHEL)")
        print()
        raise SystemExit("请先安装 FFmpeg，然后重新运行此脚本")

def install_audio_separator(has_gpu):
    """安装 audio-separator
    
    策略：先正常安装（让它拉取所有依赖），然后卸载并重装指定版本的关键包
    这样比 --no-deps 更安全，不会遗漏 audio-separator 的其他依赖
    """
    print("🎵 正在安装 audio-separator...")
    if has_gpu:
        run_pip("install", "audio-separator[gpu]")
    else:
        run_pip("install", "audio-separator")

def reinstall_critical_packages(has_gpu):
    """重装关键包到指定版本
    
    audio-separator 安装时可能会覆盖 torch/numpy 等包的版本，
    这里卸载并重装到我们需要的版本
    """
    print("🔧 正在修复关键包版本...")
    
    # 卸载被覆盖的包
    print("   卸载可能被覆盖的包...")
    run_pip("uninstall", "-y", "torch", "torchaudio", "numpy")
    
    # 重装指定版本
    if has_gpu:
        print("   重装 PyTorch (CUDA 11.8)...")
        run_pip("install", "torch==2.0.0", "torchaudio==2.0.0", 
               "--index-url", "https://download.pytorch.org/whl/cu118")
    else:
        print("   重装 PyTorch (CPU)...")
        run_pip("install", "torch==2.1.2", "torchaudio==2.1.2")
    
    # numpy 版本（根据需要指定，这里用兼容版本）
    print("   重装 numpy...")
    run_pip("install", "numpy==1.26.4")

def install_requirements():
    """安装项目依赖"""
    print("📦 正在安装项目依赖...")
    run_pip("install", "-r", "requirements.txt")

def install_spacy_model():
    """安装 spaCy 英文模型"""
    print("🔤 正在安装 spaCy 英文模型...")
    try:
        run_pip("install", "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl")
    except Exception as e:
        print(f"⚠️ spaCy 模型安装失败: {e}")
        print("   可稍后手动运行: python -m spacy download en_core_web_md")

def main():
    """主安装流程"""
    # 先安装基础依赖用于打印
    install_package("rich")
    
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    console.print(Panel(ASCII_LOGO, title="[bold cyan]VideoSubX 安装程序[/bold cyan]", 
                       border_style="cyan"))
    
    # 检测 GPU
    is_mac = platform.system() == 'Darwin'
    has_gpu = not is_mac and check_nvidia_gpu()
    
    # 安装流程
    console.print(Panel("🚀 开始安装", style="bold magenta"))
    
    # 1. 安装项目依赖
    install_requirements()
    
    # 2. 安装 audio-separator（可能会覆盖 torch/numpy 版本）
    install_audio_separator(has_gpu)
    
    # 3. 重装关键包到指定版本（修复被覆盖的 torch/numpy）
    reinstall_critical_packages(has_gpu)
    
    # 4. 安装 spaCy 模型
    install_spacy_model()
    
    # 5. 检查 FFmpeg
    check_ffmpeg()
    
    # 完成
    console.print(Panel(
        "✅ 安装完成！\\n\\n"
        "启动方式:\\n"
        "[bold cyan]python -m uvicorn main:app --host 0.0.0.0 --port 8501[/bold cyan]\\n\\n"
        "或者运行:\\n"
        "[bold cyan]run_webui.bat[/bold cyan] (Windows)",
        style="bold green"
    ))

if __name__ == "__main__":
    main()
