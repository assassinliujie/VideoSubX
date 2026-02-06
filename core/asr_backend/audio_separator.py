# audio-separator 音频分离后端
# 使用 python-audio-separator 库进行人声分离
# GitHub: https://github.com/nomadkaraoke/python-audio-separator

import os
import gc
import torch
from rich.console import Console
from rich import print as rprint
from core.utils import load_key
from core.utils.paths import _RAW_AUDIO_FILE, _VOCAL_AUDIO_FILE, _BACKGROUND_AUDIO_FILE, _AUDIO_DIR

console = Console()

def audio_separator_separate():
    """使用 audio-separator 进行音频分离"""
    
    # 只需检测 vocal.mp3 存在就跳过分离
    if os.path.exists(_VOCAL_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {_VOCAL_AUDIO_FILE} 已存在，跳过音频分离。[/yellow]")
        return
    
    os.makedirs(_AUDIO_DIR, exist_ok=True)
    
    # 延迟导入，避免未安装时报错
    try:
        from audio_separator.separator import Separator
    except ImportError:
        raise ImportError(
            "audio-separator 未安装！请运行: pip install audio-separator[gpu]"
        )
    
    # 获取配置
    model_name = load_key("audio_separator.model") or "htdemucs.yaml"
    model_cache_dir = load_key("model_dir") or "./_model_cache"
    
    console.print(f"🤖 加载 audio-separator 模型: [cyan]{model_name}[/cyan]")
    
    # 初始化分离器
    separator = Separator(
        model_file_dir=model_cache_dir,
        output_dir=_AUDIO_DIR,
        output_format="MP3",
        normalization_threshold=0.9,
        sample_rate=44100,
    )
    
    # 加载模型
    separator.load_model(model_filename=model_name)
    
    console.print("🎵 正在分离音频...")
    
    # 执行分离
    output_files = separator.separate(_RAW_AUDIO_FILE)
    
    console.print(f"[dim]分离完成，输出文件: {output_files}[/dim]")
    
    # 重命名输出文件为标准名称
    _rename_output_files(output_files, model_name)
    
    # 清理内存
    del separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 删除 raw 和 background，只保留 vocal
    if os.path.exists(_RAW_AUDIO_FILE):
        os.remove(_RAW_AUDIO_FILE)
        console.print(f"[dim]🗑️ Deleted {_RAW_AUDIO_FILE}[/dim]")
    if os.path.exists(_BACKGROUND_AUDIO_FILE):
        os.remove(_BACKGROUND_AUDIO_FILE)
        console.print(f"[dim]🗑️ Deleted {_BACKGROUND_AUDIO_FILE}[/dim]")
    
    console.print("[green]✨ 音频分离完成！[/green]")


def _rename_output_files(output_files: list, model_name: str):
    """将输出文件重命名为标准名称 (vocal.mp3, background.mp3)"""
    
    vocal_file = None
    instrumental_files = []
    
    for f in output_files:
        # audio-separator 返回的可能是相对路径或只有文件名，需要拼接 output_dir
        if not os.path.isabs(f):
            f_full = os.path.join(_AUDIO_DIR, os.path.basename(f))
        else:
            f_full = f
        
        f_lower = f.lower()
        # 判断是否为人声文件
        if 'vocal' in f_lower:
            vocal_file = f_full
        else:
            instrumental_files.append(f_full)
    
    # 重命名人声文件
    if vocal_file and vocal_file != _VOCAL_AUDIO_FILE:
        if os.path.exists(_VOCAL_AUDIO_FILE):
            os.remove(_VOCAL_AUDIO_FILE)
        os.rename(vocal_file, _VOCAL_AUDIO_FILE)
        console.print(f"🎤 人声保存至: {_VOCAL_AUDIO_FILE}")
    
    # 合并或重命名背景音乐文件
    if instrumental_files:
        if len(instrumental_files) == 1:
            # 只有一个伴奏文件，直接重命名
            if instrumental_files[0] != _BACKGROUND_AUDIO_FILE:
                if os.path.exists(_BACKGROUND_AUDIO_FILE):
                    os.remove(_BACKGROUND_AUDIO_FILE)
                os.rename(instrumental_files[0], _BACKGROUND_AUDIO_FILE)
        else:
            # 多个背景音轨，需要混合（如 Drums + Bass + Other）
            _mix_background_tracks(instrumental_files)
        
        console.print(f"🎹 背景音乐保存至: {_BACKGROUND_AUDIO_FILE}")


def _mix_background_tracks(tracks: list):
    """混合多个背景音轨为单个文件"""
    try:
        from pydub import AudioSegment
        
        mixed = None
        for track in tracks:
            audio = AudioSegment.from_file(track)
            if mixed is None:
                mixed = audio
            else:
                mixed = mixed.overlay(audio)
            # 删除临时文件
            os.remove(track)
        
        if mixed:
            mixed.export(_BACKGROUND_AUDIO_FILE, format="mp3")
    except ImportError:
        # 如果没有 pydub，就用第一个非人声文件
        rprint("[yellow]⚠️ pydub 未安装，使用第一个伴奏轨道作为背景音乐[/yellow]")
        if tracks:
            if os.path.exists(_BACKGROUND_AUDIO_FILE):
                os.remove(_BACKGROUND_AUDIO_FILE)
            os.rename(tracks[0], _BACKGROUND_AUDIO_FILE)
            # 删除其他文件
            for t in tracks[1:]:
                os.remove(t)


if __name__ == "__main__":
    audio_separator_separate()

def separate_audio():
    """统一音频分离入口"""
    audio_separator_separate()

