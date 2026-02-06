import os
import subprocess
import glob
from core.utils import *

def burn_subtitle_to_video(input_video=None, subtitle_file=None, output_file=None):
    """
    使用ffmpeg将字幕烧录到视频中
    """
    if input_video is None:
        # 查找最高质量的视频文件
        output_dir = "output"
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
        video_files = []
        
        for file in os.listdir(output_dir):
            if file.endswith(tuple(video_extensions)) and not file.startswith("output_") and "_360p." not in file:
                file_path = os.path.join(output_dir, file)
                video_files.append((file_path, os.path.getsize(file_path)))
        
        if not video_files:
            # 如果没有找到最高质量视频，使用360p
            for file in os.listdir(output_dir):
                if file.endswith(tuple(video_extensions)) and not file.startswith("output_"):
                    file_path = os.path.join(output_dir, file)
                    video_files.append((file_path, os.path.getsize(file_path)))
        
        if not video_files:
            raise ValueError("未找到视频文件")
        
        # 选择最大的文件（通常是最高质量）
        input_video = max(video_files, key=lambda x: x[1])[0]
    
    if subtitle_file is None:
        subtitle_file = "output/src_trans.ass"
        if not os.path.exists(subtitle_file):
            raise ValueError(f"字幕文件不存在: {subtitle_file}")
    
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        output_file = f"output/{base_name}_with_subtitles.mp4"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 使用ffmpeg烧录字幕
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-vf', f'ass={subtitle_file}',
        '-c:v', 'hevc_nvenc',
        '-preset', 'p4',
        '-rc', 'constqp',
        '-qp', '23',
        '-c:a', 'aac',
        '-b:a', '320k',
        '-ar', '48000',
        '-y', output_file
    ]
    
    try:
        # 移除capture_output=True以提高性能，直接输出到控制台
        subprocess.run(cmd, check=True)
        print(f"✅ 字幕烧录完成: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"❌ 字幕烧录失败")
        raise

def get_highest_quality_video():
    """
    获取最高质量的视频文件路径
    """
    output_dir = "output"
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
    video_files = []
    
    for file in os.listdir(output_dir):
        if file.endswith(tuple(video_extensions)) and not file.startswith("output_") and "_360p." not in file:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                video_files.append((file_path, os.path.getsize(file_path)))
    
    if not video_files:
        # 如果没有找到最高质量视频，返回360p视频
        for file in os.listdir(output_dir):
            if file.endswith(tuple(video_extensions)) and not file.startswith("output_"):
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path):
                    video_files.append((file_path, os.path.getsize(file_path)))
    
    if not video_files:
        return None
    
    # 选择最大的文件（通常是最高质量）
    return max(video_files, key=lambda x: x[1])[0]

def get_360p_video():
    """
    获取360p视频文件路径
    """
    output_dir = "output"
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
    
    for file in os.listdir(output_dir):
        if file.endswith(tuple(video_extensions)) and not file.startswith("output_"):
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                return file_path
    
    return None