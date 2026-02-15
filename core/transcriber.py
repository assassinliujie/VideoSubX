import os
from typing import List, Tuple

import pandas as pd

from core.asr_backend.audio_preprocess import (
    convert_video_to_audio,
    normalize_audio_volume,
    process_transcription,
    save_results,
    split_audio,
)
from core.asr_backend.audio_separator import separate_audio
from core.downloader import find_video_files
from core.utils import *
from core.utils.paths import *


def _normalize_percent_before_mfa(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """
    Normalize standalone '%' token to 'percent' before MFA alignment.
    Keep all other tokens unchanged and return replaced row indices.
    """
    if "text" not in df.columns or len(df) == 0:
        return df, []

    text_series = df["text"].astype(str).str.strip()
    percent_mask = text_series.eq("%")
    if not percent_mask.any():
        return df, []

    replaced_indices = df.index[percent_mask].tolist()
    df = df.copy()
    df.loc[percent_mask, "text"] = "percent"
    rprint(
        f"[cyan]Normalized {int(percent_mask.sum())} '%' token(s) to 'percent' before MFA.[/cyan]"
    )
    return df, replaced_indices


def _restore_percent_after_mfa(df: pd.DataFrame, replaced_indices: List[int]) -> pd.DataFrame:
    """Restore rows replaced before MFA back to '%' by recorded indices only."""
    if not replaced_indices or "text" not in df.columns or len(df) == 0:
        return df

    valid_indices = [i for i in replaced_indices if 0 <= int(i) < len(df)]
    if not valid_indices:
        return df

    df = df.copy()
    df.loc[valid_indices, "text"] = "%"
    rprint(f"[cyan]Restored {len(valid_indices)} token(s) from 'percent' back to '%' after MFA.[/cyan]")
    return df


def _resolve_majority_language(results: List[dict]) -> str:
    """Resolve the most frequent non-empty language from segment-level ASR results."""
    lang_counts = {}
    for result in results:
        lang = str(result.get("language", "")).strip().lower()
        if not lang:
            continue
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    if not lang_counts:
        return ""
    return max(lang_counts.items(), key=lambda kv: kv[1])[0]


def _resolve_alignment_mode() -> Tuple[int, str]:
    """
    Resolve alignment mode from numeric selector in config:
      alignment.mode = 1 | 2 | 3 | 4
      1: raw
      2: stable
      3: mfa
      4: stable_mfa
    """
    try:
        raw_mode = load_key("alignment.mode")
    except Exception as e:
        raise KeyError("Missing config key: alignment.mode (use 1/2/3/4).") from e

    if isinstance(raw_mode, bool):
        raise ValueError(
            f"Invalid alignment.mode: {raw_mode}. Use integer selector 1/2/3/4."
        )

    if isinstance(raw_mode, int):
        mode_selector = raw_mode
    else:
        mode_text = str(raw_mode).strip()
        if not mode_text.isdigit():
            raise ValueError(
                f"Invalid alignment.mode: {raw_mode}. Use integer selector 1/2/3/4."
            )
        mode_selector = int(mode_text)

    mode_map = {
        1: "raw",
        2: "stable",
        3: "mfa",
        4: "stable_mfa",
    }
    if mode_selector not in mode_map:
        raise ValueError(
            f"Invalid alignment.mode: {mode_selector}. Allowed selectors: 1(raw), 2(stable), 3(mfa), 4(stable_mfa)."
        )

    return mode_selector, mode_map[mode_selector]


@check_file_exists(_2_CLEANED_CHUNKS)
def transcribe():
    # 1) Ensure source audio exists
    if not os.path.exists(_VOCAL_AUDIO_FILE):
        video_file = find_video_files()
        convert_video_to_audio(video_file)

    # 2) Always use audio-separator and then normalize vocals
    separate_audio()
    vocal_audio = normalize_audio_volume(_VOCAL_AUDIO_FILE, _VOCAL_AUDIO_FILE, format="mp3")

    # 3) Split vocal track into ASR segments
    segments = split_audio(vocal_audio)

    runtime = load_key("whisper.runtime")
    if runtime != "stable":
        raise ValueError(f"Unsupported ASR runtime: {runtime}. Only 'stable' is supported.")

    from core.asr_backend.stable_ts import align_words_with_stable, release_model, transcribe_audio_stable

    rprint("[cyan]Transcribing audio with stable-ts...[/cyan]")
    mode_selector, alignment_mode = _resolve_alignment_mode()
    rprint(f"[cyan][Experimental] Alignment mode:[/cyan] {mode_selector} ({alignment_mode})")
    whisper_language = str(load_key("whisper.language") or "").strip().lower()

    # 4) Reuse loaded stable-ts model across all segments, then release once
    all_results = []
    combined_result = {"segments": []}
    locked_language = ""
    try:
        for idx, (start, end) in enumerate(segments):
            forced_language = locked_language or None
            result = transcribe_audio_stable(vocal_audio, start, end, forced_language=forced_language)
            all_results.append(result)

            # If whisper.language=auto, detect once on the first usable segment and lock it for all following segments.
            if whisper_language == "auto" and not locked_language:
                detected = str(result.get("language", "")).strip().lower()
                if detected and detected != "auto":
                    locked_language = detected
                    update_key("whisper.detected_language", locked_language)
                    rprint(
                        f"[cyan]Locked ASR language from segment {idx + 1}:[/cyan] {locked_language}"
                    )

        # 5) Merge segment-level ASR outputs
        for result in all_results:
            combined_result["segments"].extend(result["segments"])
        if all_results:
            merged_language = locked_language
            if not merged_language and whisper_language and whisper_language != "auto":
                merged_language = whisper_language
            if not merged_language:
                merged_language = _resolve_majority_language(all_results)
            if merged_language:
                combined_result["language"] = merged_language
                update_key("whisper.detected_language", merged_language)
                rprint(f"[cyan]Resolved transcription language:[/cyan] {merged_language}")

        # 6) Alignment stage (pre-MFA)
        if alignment_mode in {"stable", "stable_mfa"}:
            rprint("[cyan][Experimental] Running stable-ts align_words...[/cyan]")
            combined_result = align_words_with_stable(vocal_audio, combined_result)
        elif alignment_mode == "raw":
            rprint("[yellow][Experimental] raw mode: skip all alignment.[/yellow]")
        elif alignment_mode == "mfa":
            rprint("[cyan][Experimental] mfa mode: skip stable-ts align_words.[/cyan]")
    finally:
        release_model()

    # 7) Post-process transcription
    df = process_transcription(combined_result)

    # 8) Optional MFA forced alignment
    if alignment_mode in {"mfa", "stable_mfa"}:
        from core.asr_backend.mfa_aligner import align_transcription

        df, replaced_indices = _normalize_percent_before_mfa(df)
        rprint("[cyan][Experimental] Running MFA alignment...[/cyan]")
        df = align_transcription(df, vocal_audio)
        df = _restore_percent_after_mfa(df, replaced_indices)

    # 9) Save cleaned chunks
    save_results(df)


if __name__ == "__main__":
    transcribe()
