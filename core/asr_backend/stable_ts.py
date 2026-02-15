import os
import subprocess
import time
import warnings
from typing import Dict, Optional

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


def _count_overlong_words(result_dict: Dict, max_len: int = 30):
    total = 0
    overlong = 0
    for segment in result_dict.get("segments", []):
        for word in segment.get("words", []):
            total += 1
            text = str(word.get("word", "")).strip()
            if len(text) > max_len:
                overlong += 1
    return total, overlong


def _is_alignment_degraded(original_result: Dict, aligned_result: Dict):
    orig_total, orig_overlong = _count_overlong_words(original_result, max_len=30)
    aligned_total, aligned_overlong = _count_overlong_words(aligned_result, max_len=30)

    if aligned_total == 0:
        return True, f"aligned words are empty (orig={orig_total}, aligned=0)"

    # Word count should stay close after timestamp-only refinement.
    if orig_total > 0 and aligned_total < int(orig_total * 0.85):
        return True, f"word count dropped too much (orig={orig_total}, aligned={aligned_total})"
    if orig_total > 0 and aligned_total > int(orig_total * 1.20):
        return True, f"word count increased too much (orig={orig_total}, aligned={aligned_total})"

    # Overlong words should not spike after alignment.
    overlong_spike_limit = max(orig_overlong + 2, int(aligned_total * 0.02))
    if aligned_overlong > overlong_spike_limit:
        return (
            True,
            f"overlong words spiked (orig={orig_overlong}, aligned={aligned_overlong}, limit={overlong_spike_limit})",
        )

    return False, ""


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


def align_words_with_stable(vocal_audio_file: str, result_dict: Dict) -> Dict:
    """
    Use stable-ts align_words() as a lightweight post-ASR timestamp refinement.
    If alignment fails, return the original result unchanged.
    """
    if not result_dict or "segments" not in result_dict or not result_dict["segments"]:
        return result_dict

    language_arg = None

    # 1) Prefer language carried by ASR result
    result_language = str(result_dict.get("language", "")).strip().lower()
    if result_language and result_language != "auto":
        language_arg = result_language

    # 2) Fall back to detected language
    if not language_arg:
        detected_language = str(load_key("whisper.detected_language") or "").strip().lower()
        if detected_language and detected_language != "auto":
            language_arg = detected_language

    # 3) Finally, use explicit whisper.language when not auto
    if not language_arg:
        whisper_language = str(load_key("whisper.language") or "").strip().lower()
        if whisper_language and "auto" not in whisper_language:
            language_arg = whisper_language

    if not language_arg:
        rprint("[yellow]stable-ts align_words skipped: language is unavailable.[/yellow]")
        return result_dict

    rprint(f"[cyan]stable-ts align_words language:[/cyan] {language_arg}")
    model = _get_or_load_model()
    align_start_time = time.time()

    # IMPORTANT:
    # stable-ts WhisperResult(Segment(words=...)) concatenates word.word directly.
    # If upstream words are stripped (no leading spaces), text can collapse into "Comeon.Iam...".
    # For align_words we pass text-only segments to preserve original spacing semantics.
    align_input_segments = []
    for seg in result_dict.get("segments", []):
        text = str(seg.get("text", "") or "")
        if not text and seg.get("words"):
            # Fallback only when text is missing: rebuild readable text with explicit spaces.
            rebuilt = [str(w.get("word", "")).strip() for w in seg.get("words", []) if str(w.get("word", "")).strip()]
            text = " ".join(rebuilt)
        align_input_segments.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text.strip(),
            }
        )

    try:
        aligned_result = model.align_words(
            vocal_audio_file,
            align_input_segments,
            language=language_arg,
            vad=True,
            vad_threshold=0.35,
            min_word_dur=0.1,
            suppress_silence=True,
            only_voice_freq=True,
            use_word_position=True,
        )
    except Exception as e:
        rprint(f"[yellow]stable-ts align_words failed, keep original timestamps: {e}[/yellow]")
        return result_dict

    align_time = time.time() - align_start_time
    rprint(f"[cyan]stable-ts align_words time:[/cyan] {align_time:.2f}s")

    if hasattr(aligned_result, "to_dict"):
        aligned_dict = aligned_result.to_dict()
    elif isinstance(aligned_result, dict):
        aligned_dict = aligned_result
    elif isinstance(aligned_result, list):
        aligned_dict = {"segments": aligned_result}
    else:
        rprint("[yellow]Unexpected align_words output, keep original timestamps.[/yellow]")
        return result_dict

    if "segments" not in aligned_dict:
        rprint("[yellow]align_words output missing segments, keep original timestamps.[/yellow]")
        return result_dict

    for segment in aligned_dict["segments"]:
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
            cleaned_words.append(word)

        segment["words"] = cleaned_words

    if "language" not in aligned_dict and "language" in result_dict:
        aligned_dict["language"] = result_dict["language"]

    degraded, reason = _is_alignment_degraded(result_dict, aligned_dict)
    if degraded:
        rprint(f"[yellow]stable-ts align_words output degraded ({reason}), keep original timestamps.[/yellow]")
        return result_dict

    return aligned_dict


@except_handler("stable-ts processing error:")
def transcribe_audio_stable(vocal_audio_file, start, end, forced_language: Optional[str] = None):
    whisper_language = str(load_key("whisper.language") or "").strip().lower()
    model = _get_or_load_model()

    audio_segment, _ = librosa.load(vocal_audio_file, sr=16000, offset=start, duration=end - start, mono=True)

    transcribe_start_time = time.time()
    language_arg = str(forced_language or whisper_language or "").strip().lower()
    if not language_arg or "auto" in language_arg:
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

    return result_dict
