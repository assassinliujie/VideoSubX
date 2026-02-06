import os
from core.utils import *
from core.asr_backend.audio_separator import separate_audio
from core.asr_backend.audio_preprocess import process_transcription, convert_video_to_audio, split_audio, save_results, normalize_audio_volume
from core.downloader import find_video_files
from core.utils.paths import *

@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1. 检查是否需要提取音频（如果 vocal.mp3 已存在则跳过）
    if not os.path.exists(_VOCAL_AUDIO_FILE):
        video_file = find_video_files()
        convert_video_to_audio(video_file)
    
    # 2. 音频分离（人声/背景）:
    if load_key("demucs"):
        separate_audio()
        vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")
    else:
        # 如果没开 demucs 但 vocal 存在，直接用 vocal
        if os.path.exists(_VOCAL_AUDIO_FILE):
            vocal_audio = _VOCAL_AUDIO_FILE
        else:
            vocal_audio = _VOCAL_AUDIO_FILE

    # 3. 用人声文件检测语音边界（避免片头背景音乐干扰）
    segments = split_audio(vocal_audio)
    
    # 4. 转录音频片段
    all_results = []
    runtime = load_key("whisper.runtime")
    if runtime != "stable":
        raise ValueError(f"Unsupported ASR runtime: {runtime}. Only 'stable' is supported.")
    
    from core.asr_backend.stable_ts import transcribe_audio_stable as ts
    rprint("[cyan]🎤 Transcribing audio with stable-ts...[/cyan]")

    for start, end in segments:
        # 只使用 vocal_audio 进行 ASR，不再传 raw
        result = ts(vocal_audio, start, end)
        all_results.append(result)
    
    # 5. 合并结果
    combined_result = {'segments': []}
    for result in all_results:
        combined_result['segments'].extend(result['segments'])
    
    # 6. 处理数据
    df = process_transcription(combined_result)
    save_results(df)
        
if __name__ == "__main__":
    transcribe()