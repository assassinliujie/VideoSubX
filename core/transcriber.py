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

    from core.asr_backend.stable_ts import release_model, transcribe_audio_stable

    rprint("[cyan]Transcribing audio with stable-ts...[/cyan]")

    # 4) Reuse loaded stable-ts model across all segments, then release once
    all_results = []
    try:
        for start, end in segments:
            result = transcribe_audio_stable(vocal_audio, start, end)
            all_results.append(result)
    finally:
        release_model()

    # 5) Merge segment-level ASR outputs
    combined_result = {"segments": []}
    for result in all_results:
        combined_result["segments"].extend(result["segments"])

    # 6) Post-process transcription
    df = process_transcription(combined_result)

    # 7) Optional MFA forced alignment
    if load_key("mfa.enabled"):
        from core.asr_backend.mfa_aligner import align_transcription

        df, replaced_indices = _normalize_percent_before_mfa(df)
        rprint("[cyan][Experimental] Running MFA alignment...[/cyan]")
        df = align_transcription(df, vocal_audio)
        df = _restore_percent_after_mfa(df, replaced_indices)

    # 8) Save cleaned chunks
    save_results(df)


if __name__ == "__main__":
    transcribe()
