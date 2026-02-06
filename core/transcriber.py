import os

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

        rprint("[cyan][Experimental] Running MFA alignment...[/cyan]")
        df = align_transcription(df, vocal_audio)

    # 8) Save cleaned chunks
    save_results(df)


if __name__ == "__main__":
    transcribe()
