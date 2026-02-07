import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd

from core.prompts import get_english_correction_prompt
from core.utils import ask_gpt, load_key, rprint
from core.utils.paths import _2_CLEANED_CHUNKS

_COLLOQUIAL_FORMS = {
    "gonna",
    "wanna",
    "gotta",
    "kinda",
    "sorta",
    "ain't",
    "y'all",
}


def _load_key_or_default(key, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _load_bool_key(key, default=False):
    value = _load_key_or_default(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _start_key(value) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _normalize_word(text: str) -> str:
    return str(text).strip().strip('"').strip()


def _build_tokens(df: pd.DataFrame) -> List[Dict]:
    tokens = []
    for _, row in df.iterrows():
        word = _normalize_word(row["text"])
        if not word:
            continue
        tokens.append(
            {
                "start_key": _start_key(row["start"]),
                "start": float(row["start"]),
                "word": word,
            }
        )
    return tokens


def _get_correction_api_settings() -> Dict:
    # Empty value means fallback to global api.* settings.
    return {
        "key": _load_key_or_default("english_correction.api.key", ""),
        "base_url": _load_key_or_default("english_correction.api.base_url", ""),
        "model": _load_key_or_default("english_correction.api.model", ""),
        "llm_support_json": _load_key_or_default("english_correction.api.llm_support_json", True),
        "request_timeout_sec": _load_key_or_default("english_correction.api.request_timeout_sec", None),
        "request_retries": _load_key_or_default("english_correction.api.request_retries", None),
        "request_retry_delay_sec": _load_key_or_default("english_correction.api.request_retry_delay_sec", None),
    }


def _valid_correction_response(response_data):
    if "corrections" not in response_data:
        return {"status": "error", "message": "Missing required key: `corrections`"}
    if not isinstance(response_data["corrections"], list):
        return {"status": "error", "message": "`corrections` must be a list"}

    for i, item in enumerate(response_data["corrections"]):
        if not isinstance(item, dict):
            return {"status": "error", "message": f"`corrections[{i}]` must be an object"}
        for key in ("start_key", "source", "target"):
            if key not in item:
                return {"status": "error", "message": f"Missing key in `corrections[{i}]`: `{key}`"}
    return {"status": "success", "message": "English correction response validated"}


def _apply_corrections(df: pd.DataFrame, corrections: List[Dict], run_id: str):
    start_to_indices = {}
    for idx, row in df.iterrows():
        key = _start_key(row["start"])
        start_to_indices.setdefault(key, []).append(idx)

    used_indices = set()
    applied_count = 0
    audit_rows = []
    for item in corrections:
        start_key = str(item.get("start_key", "")).strip()
        source = _normalize_word(item.get("source", ""))
        target = _normalize_word(item.get("target", ""))
        confidence = str(item.get("confidence", "")).strip().lower()
        reason = str(item.get("reason", "")).strip()
        corr_type = str(item.get("type", "")).strip()

        record = {
            "run_id": run_id,
            "status": "skipped",
            "skip_reason": "",
            "start_key": start_key,
            "source": source,
            "target": target,
            "confidence": confidence,
            "type": corr_type,
            "reason": reason,
            "row_index": "",
            "row_start": "",
            "before": "",
            "after": "",
        }

        if not start_key or not source or not target:
            record["skip_reason"] = "missing_required_fields"
            audit_rows.append(record)
            continue
        if source == target:
            record["skip_reason"] = "source_equals_target"
            audit_rows.append(record)
            continue
        if " " in source or " " in target:
            # Enforce token-level replacement only.
            record["skip_reason"] = "non_token_replacement"
            audit_rows.append(record)
            continue
        if source.lower() in _COLLOQUIAL_FORMS:
            # Hard guard: do not normalize spoken colloquial forms.
            record["skip_reason"] = "colloquial_form_guard"
            audit_rows.append(record)
            continue
        # If confidence is provided, apply only high confidence corrections.
        if confidence and confidence not in {"high", "very_high", "very high"}:
            record["skip_reason"] = "low_confidence"
            audit_rows.append(record)
            continue

        candidates = start_to_indices.get(start_key, [])
        idx = next((x for x in candidates if x not in used_indices), None)
        if idx is None:
            record["skip_reason"] = "start_key_not_found_or_already_used"
            audit_rows.append(record)
            continue

        current = _normalize_word(df.at[idx, "text"])
        if current != source:
            # Safety gate: only replace exact source token.
            record["skip_reason"] = "source_mismatch_with_current_token"
            record["row_index"] = int(idx)
            record["row_start"] = _start_key(df.at[idx, "start"])
            record["before"] = current
            audit_rows.append(record)
            continue

        df.at[idx, "text"] = f'"{target}"'
        used_indices.add(idx)
        applied_count += 1
        record["status"] = "applied"
        record["row_index"] = int(idx)
        record["row_start"] = _start_key(df.at[idx, "start"])
        record["before"] = current
        record["after"] = target
        audit_rows.append(record)

    return applied_count, audit_rows


def _write_changelog(rows: List[Dict]):
    if not rows:
        return None
    log_path = "output/log/english_correction_changelog.csv"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    df_log = pd.DataFrame(rows)
    df_log.insert(0, "logged_at", datetime.now().isoformat(timespec="seconds"))
    file_exists = os.path.exists(log_path)
    df_log.to_csv(log_path, mode="a", header=not file_exists, index=False, encoding="utf-8-sig")
    return log_path


def correct_english_asr_tokens():
    if not _load_bool_key("english_correction.enabled", False):
        rprint("[dim]English correction disabled, skip.[/dim]")
        return

    only_when_en = _load_bool_key("english_correction.only_when_detected_language", True)
    detected_language = _load_key_or_default("whisper.detected_language", "")
    if only_when_en and str(detected_language).lower() != "en":
        rprint(f"[dim]English correction skipped (detected_language={detected_language}).[/dim]")
        return

    if not os.path.exists(_2_CLEANED_CHUNKS):
        raise FileNotFoundError(f"Missing input file: {_2_CLEANED_CHUNKS}")

    df = pd.read_excel(_2_CLEANED_CHUNKS)
    if len(df) == 0:
        rprint("[dim]English correction skipped (no rows).[/dim]")
        return

    tokens = _build_tokens(df)
    if not tokens:
        rprint("[dim]English correction skipped (no valid tokens).[/dim]")
        return

    tokens_json = json.dumps(tokens, ensure_ascii=False, indent=2)
    prompt = get_english_correction_prompt(tokens_json)
    rprint(f"[cyan]Running English ASR correction on {len(tokens)} tokens...[/cyan]")

    response = ask_gpt(
        prompt,
        resp_type="json",
        valid_def=_valid_correction_response,
        log_title="english_correction",
        api_settings=_get_correction_api_settings(),
    )
    corrections = response.get("corrections", [])
    if not corrections:
        rprint("[green]No English ASR corrections suggested.[/green]")
        return

    backup_path = "output/log/cleaned_chunks_before_english_correction.xlsx"
    if not os.path.exists(backup_path):
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        df.to_excel(backup_path, index=False)
        rprint(f"[dim]Backup created: {backup_path}[/dim]")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    applied, audit_rows = _apply_corrections(df, corrections, run_id=run_id)
    log_path = _write_changelog(audit_rows)
    if log_path:
        rprint(f"[dim]English correction changelog: {log_path} (run_id={run_id})[/dim]")

    if applied == 0:
        rprint("[yellow]English corrections returned, but no safe replacements were applied.[/yellow]")
        return

    df.to_excel(_2_CLEANED_CHUNKS, index=False)
    rprint(f"[green]Applied {applied} English token corrections to {_2_CLEANED_CHUNKS}[/green]")


if __name__ == "__main__":
    correct_english_asr_tokens()
