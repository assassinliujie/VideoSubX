import os
import warnings
import time
import subprocess
import torch
import stable_whisper
import librosa
from rich import print as rprint
from core.utils import *

warnings.filterwarnings("ignore")
MODEL_DIR = load_key("model_dir")

@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start
        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")
    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")
    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url

@except_handler("stable-ts processing error:")
def transcribe_audio_stable(vocal_audio_file, start, end):
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting stable-ts using device: {device} ...")
    
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        rprint(f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB")
    
    # 加载模型
    if WHISPER_LANGUAGE == 'zh':
        model_name = "Belle-whisper-large-v3-zh-punct"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)
    
    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local stable-ts model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using stable-ts model from HuggingFace:[/green] {model_name} ...")
    
    model = stable_whisper.load_model(model_name, device=device, download_root=MODEL_DIR)
    
    def load_audio_segment(audio_file, start, end):
        audio, _ = librosa.load(audio_file, sr=16000, offset=start, duration=end - start, mono=True)
        return audio
    
    # Use vocal_audio_file if provided (for Demucs processed audio), otherwise it falls back to raw_audio_file in _2_asr.py
    audio_segment = load_audio_segment(vocal_audio_file, start, end)
    
    # 转录并获取词级时间戳
    transcribe_start_time = time.time()
    rprint("[bold green]Note: You will see Progress if working correctly ↓[/bold green]")
    
    # 安全处理语言参数
    language_arg = WHISPER_LANGUAGE
    if language_arg and 'auto' in language_arg:
        language_arg = None

    result = model.transcribe(
        audio_segment,
        language=language_arg,
        word_timestamps=True,
        verbose=False,
        regroup=False,  # 禁用自动重组，保持更长的句子
        vad=True,       # 启用VAD辅助定位，修复句尾时间戳漂移
        vad_threshold=0.35, # 恢复默认 VAD 阈值
        min_word_dur=0.1,   # 恢复默认
        suppress_silence=True,  # 显式开启静音抑制
        only_voice_freq=True,   # 只保留人声频率(200-5000Hz)，过滤底噪
        use_word_position=True #  恢复默认
    )
    
    #rprint("[cyan]🔧 Refining timestamps...[/cyan]")
    #model.refine(audio_segment, result) # 禁用慢速Refine
    
    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {transcribe_time:.2f}s")
    
    # 转换为字典格式
    result_dict = result.to_dict()
    
    # 保存检测到的语言
    update_key("whisper.detected_language", result_dict['language'])
    # 只有当用户明确指定了非中文语言（不是 auto）时，才报错提示
    if result_dict['language'] == 'zh' and WHISPER_LANGUAGE != 'zh' and 'auto' not in WHISPER_LANGUAGE:
        raise ValueError("Please specify the transcription language as zh and try again!")
    
    # 调整时间戳（加上起始偏移）并清理空格
    for segment in result_dict['segments']:
        segment['start'] += start
        segment['end'] += start
        segment['text'] = segment['text'].strip()
        
        # 确保words字段存在
        if 'words' not in segment:
            segment['words'] = []
        
        # 清理每个单词的前后空格并重新构建文本
        cleaned_words = []
        for word in segment.get('words', []):
            if 'word' in word:
                word['word'] = word['word'].strip()
                if word['word']:  # 只保留非空单词
                    if 'start' in word:
                        word['start'] += start
                    if 'end' in word:
                        word['end'] += start
                    cleaned_words.append(word)
        segment['words'] = cleaned_words
        
        # 重新构建segment文本，确保单词间只有一个空格
        if cleaned_words:
            segment['text'] = ' '.join([w['word'] for w in cleaned_words])
    
    # 清理GPU资源
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result_dict