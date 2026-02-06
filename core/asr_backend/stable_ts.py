import os
import subprocess
import time
import warnings

import librosa
import stable_whisper
import torch
from rich import print as rprint

from core.utils import *

warnings.filterwarnings("ignore")
MODEL_DIR = load_key("model_dir")

_MODEL = None
_MODEL_DEVICE = None
_MODEL_SOURCE = None


@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {"Official": "huggingface.co", "Mirror": "hf-mirror.com"}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float("inf")

    rprint("[cyan]Checking HuggingFace mirrors...[/cyan]")
    for name, domain in mirrors.items():
        if os.name == "nt":
            cmd = ["ping", "-n", "1", "-w", "3000", domain]
        else:
            cmd = ["ping", "-c", "1", "-W", "3", domain]

        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        response_time = time.time() - start

        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]{name}:[/green] {response_time:.2f}s")

    if best_time == float("inf"):
        rprint("[yellow]All mirrors failed, using default[/yellow]")

    rprint(f"[cyan]Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url


def _resolve_model_source(whisper_language: str):
    if whisper_language == "zh":
        model_name = "Belle-whisper-large-v3-zh-punct"
        local_model = os.path.join(MODEL_DIR, model_name)
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)

    if os.path.exists(local_model):
        rprint(f"[green]Loading local stable-ts model:[/green] {local_model}")
        return local_model

    rprint(f"[green]Using stable-ts model from HuggingFace:[/green] {model_name}")
    return model_name


def _get_or_load_model():
    global _MODEL, _MODEL_DEVICE, _MODEL_SOURCE

    whisper_language = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = _resolve_model_source(whisper_language)

    if _MODEL is not None and _MODEL_DEVICE == device and _MODEL_SOURCE == source:
        return _MODEL

    # Reload if model/device/source changed
    release_model()

    mirror = check_hf_mirror()
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror

    rprint(f"[cyan]Loading stable-ts model on device: {device}[/cyan]")
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        rprint(f"[cyan]GPU memory:[/cyan] {gpu_mem:.2f} GB")

    _MODEL = stable_whisper.load_model(source, device=device, download_root=MODEL_DIR)
    _MODEL_DEVICE = device
    _MODEL_SOURCE = source
    return _MODEL


def release_model():
    global _MODEL, _MODEL_DEVICE, _MODEL_SOURCE

    if _MODEL is not None:
        del _MODEL
        _MODEL = None
        _MODEL_DEVICE = None
        _MODEL_SOURCE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rprint("[dim]stable-ts model released.[/dim]")


@except_handler("stable-ts processing error:")
def transcribe_audio_stable(vocal_audio_file, start, end):
    whisper_language = load_key("whisper.language")
    model = _get_or_load_model()

    audio_segment, _ = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=end - start, mono=True)

    transcribe_start_time = time.time()
    language_arg = whisper_language
    if language_arg and "auto" in language_arg:
        language_arg = None

    result = model.transcribe(
        audio_segment,
        language=language_arg,
        word_timestamps=True,
        verbose=False,
        regroup=False,
        vad=True,
        vad_threshold=0.35,
        min_word_dur=0.1,
        suppress_silence=True,
        only_voice_freq=True,
        use_word_position=True,
    )

    transcribe_time = time.time() - transcribe_start_time
    rprint(f"[cyan]Transcribe segment time:[/cyan] {transcribe_time:.2f}s")

    result_dict = result.to_dict()
    update_key("whisper.detected_language", result_dict["language"])

    if result_dict["language"] == "zh" and whisper_language != "zh" and "auto" not in whisper_language:
        raise ValueError("Please specify the transcription language as zh and try again!")

    for segment in result_dict["segments"]:
        segment["start"] += start
        segment["end"] += start
        segment["text"] = segment.get("text", "").strip()

        if "words" not in segment:
            segment["words"] = []

        cleaned_words = []
        for word in segment.get("words", []):
            if "word" not in word:
                continue
            word["word"] = word["word"].strip()
            if not word["word"]:
                continue

            if "start" in word:
                word["start"] += start
            if "end" in word:
                word["end"] += start
            cleaned_words.append(word)

        segment["words"] = cleaned_words
        if cleaned_words:
            segment["text"] = " ".join([w["word"] for w in cleaned_words])

    return result_dict
