import pandas as pd
import json
import concurrent.futures
import re
import time
import os
import hashlib
import requests
from core.translate_lines import translate_lines
from core.prompts import get_prompt_single_pass_full_polish
from core.summarizer import search_things_to_note_in_prompt
from core.utils.text_trim import check_len_then_trim
from core.subtitle_generator import align_timestamp
from core.utils import *
from core.utils.ask_gpt import (
    _is_claude_model,
    _load_key_or_default,
    _normalize_claude_messages_url,
    _normalize_openai_base_url,
    _pick_setting,
    _save_cache,
    _to_float,
    _to_int,
)
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.paths import *
console = Console()

POLISH_LINE_PATTERN = re.compile(r"^\s*\[(\d+)\]\s*:?[ \t]*(.*?)\s*$")
POLISH_PROGRESS_REPORT_STEP = 20
POLISH_PROGRESS_STATE_VERSION = 1
POLISH_TAIL_PLACEHOLDER = "拜拜"
POLISH_STREAM_SNAPSHOT_INTERVAL_SEC = 0.5

# 拆分文本块的函数
def split_chunks_by_chars(chunk_size, max_i): 
    """根据字符数将文本拆分为块，返回多行文本块列表"""
    with open(_3_2_SPLIT_BY_MEANING, "r", encoding="utf-8") as file:
        sentences = file.read().strip().split('\n')

    chunks = []
    chunk = ''
    sentence_count = 0
    for sentence in sentences:
        if len(chunk) + len(sentence + '\n') > chunk_size or sentence_count == max_i:
            chunks.append(chunk.strip())
            chunk = sentence + '\n'
            sentence_count = 1
        else:
            chunk += sentence + '\n'
            sentence_count += 1
    chunks.append(chunk.strip())
    return chunks

# 获取相邻块的上下文
def get_previous_content(chunks, chunk_index):
    return None if chunk_index == 0 else chunks[chunk_index - 1].split('\n')[-8:] # 获取最后8行作为上下文
def get_after_content(chunks, chunk_index):
    return None if chunk_index == len(chunks) - 1 else chunks[chunk_index + 1].split('\n')[:8] # 获取前8行作为上下文

# 🔍 翻译单个块
def translate_chunk(chunk, chunks, theme_prompt, i):
    things_to_note_prompt = search_things_to_note_in_prompt(chunk)
    previous_content_prompt = get_previous_content(chunks, i)
    after_content_prompt = get_after_content(chunks, i)
    translation, english_result = translate_lines(chunk, previous_content_prompt, after_content_prompt, things_to_note_prompt, theme_prompt, i)
    return i, english_result, translation

# 计算相似度函数
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def _load_single_pass_full_polish_api_settings():
    defaults = {
        "key": "",
        "base_url": "",
        "model": "",
        "llm_support_json": True,
        "request_timeout_sec": 120,
        "request_retries": 2,
        "request_retry_delay_sec": 1,
    }
    resolved = {}
    for k, v in defaults.items():
        try:
            resolved[k] = load_key(f"single_pass_full_polish.api.{k}")
        except Exception:
            resolved[k] = v
    return resolved

def _sanitize_single_line_text(value):
    return str(value).replace("\n", " ").replace("\r", " ").strip()

def _hash_lines(lines):
    payload = "\n".join(_sanitize_single_line_text(line) for line in lines)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _hash_text(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()

def _atomic_write_json(file_path, payload):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, file_path)

def _atomic_write_text(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    os.replace(tmp_path, file_path)

def _load_single_pass_full_polish_progress_state():
    if not os.path.exists(_4_2_SINGLE_PASS_FULL_POLISH_PROGRESS):
        return None

    with open(_4_2_SINGLE_PASS_FULL_POLISH_PROGRESS, "r", encoding="utf-8") as f:
        return json.load(f)

def _render_single_pass_full_polish_preview(
    *,
    prefix_lines,
    line_count,
    status,
    final_line_marker_seen,
    completion_reason,
):
    committed_count = len(prefix_lines)
    next_line = min(committed_count + 1, line_count + 1)
    header_lines = [
        f"# status: {status}",
        f"# committed: {committed_count}/{line_count}",
        f"# next_line: {next_line}",
        f"# final_line_marker_seen: {str(bool(final_line_marker_seen)).lower()}",
        f"# completion_reason: {completion_reason or 'none'}",
        "# confirmed_lines:",
    ]
    body_lines = [
        f"[{i}] {_sanitize_single_line_text(line)}"
        for i, line in enumerate(prefix_lines, 1)
    ]
    return "\n".join(header_lines + body_lines)

def _persist_single_pass_full_polish_stream_snapshot(
    *,
    start_line,
    line_count,
    committed_count,
    final_line_marker_seen,
    raw_content,
):
    header_lines = [
        f"# start_line: {start_line}",
        f"# committed_in_this_round: {committed_count}",
        f"# target_end_line: {line_count}",
        f"# final_line_marker_seen: {str(bool(final_line_marker_seen)).lower()}",
        "# raw_stream:",
    ]
    content = "\n".join(header_lines)
    if raw_content:
        content += "\n" + raw_content
    _atomic_write_text(_4_2_SINGLE_PASS_FULL_POLISH_STREAM, content)

def _has_seen_target_line_marker(raw_content, target_line):
    if not raw_content or target_line < 1:
        return False
    pattern = re.compile(
        rf"(?:^|\n)[ \t]*\[{target_line}\](?=[ \t:：]|$)",
        re.MULTILINE,
    )
    return bool(pattern.search(raw_content))

def _persist_single_pass_full_polish_progress_state(
    *,
    src_lines,
    draft_lines,
    polished_prefix_lines,
    summary_prompt,
    status="in_progress",
    final_line_marker_seen=False,
    completion_reason=None,
):
    polished_prefix_lines = [
        _sanitize_single_line_text(line) for line in polished_prefix_lines
    ]
    draft_lines = [_sanitize_single_line_text(line) for line in draft_lines]
    state = {
        "version": POLISH_PROGRESS_STATE_VERSION,
        "status": status,
        "line_count": len(src_lines),
        "source_hash": _hash_lines(src_lines),
        "draft_hash": _hash_lines(draft_lines),
        "summary_hash": _hash_text(summary_prompt),
        "updated_at": int(time.time()),
        "committed_count": len(polished_prefix_lines),
        "base_draft_lines": draft_lines,
        "polished_prefix_lines": polished_prefix_lines,
        "final_line_marker_seen": bool(final_line_marker_seen),
        "completion_reason": completion_reason,
    }
    _atomic_write_json(_4_2_SINGLE_PASS_FULL_POLISH_PROGRESS, state)
    _atomic_write_text(
        _4_2_SINGLE_PASS_FULL_POLISH_PREVIEW,
        _render_single_pass_full_polish_preview(
            prefix_lines=polished_prefix_lines,
            line_count=len(src_lines),
            status=status,
            final_line_marker_seen=final_line_marker_seen,
            completion_reason=completion_reason,
        ),
    )

def _cleanup_single_pass_full_polish_progress():
    for file_path in (
        _4_2_SINGLE_PASS_FULL_POLISH_PROGRESS,
        _4_2_SINGLE_PASS_FULL_POLISH_PREVIEW,
        _4_2_SINGLE_PASS_FULL_POLISH_STREAM,
    ):
        if os.path.exists(file_path):
            os.remove(file_path)

def _force_complete_single_pass_full_polish_tail(
    *,
    src_lines,
    draft_lines,
    polished_lines,
    summary_prompt,
    start_line,
):
    for i in range(start_line - 1, len(polished_lines)):
        polished_lines[i] = POLISH_TAIL_PLACEHOLDER

    _persist_single_pass_full_polish_progress_state(
        src_lines=src_lines,
        draft_lines=draft_lines,
        polished_prefix_lines=polished_lines,
        summary_prompt=summary_prompt,
        status="completed",
        final_line_marker_seen=True,
        completion_reason="forced_tail_placeholder",
    )
    console.print(
        f"[yellow]Final line marker detected earlier. "
        f"Filled lines {start_line}-{len(polished_lines)} with placeholder `{POLISH_TAIL_PLACEHOLDER}` "
        "and marked full-text polish as completed.[/yellow]"
    )
    return polished_lines

def _restore_single_pass_full_polish_progress(src_lines, draft_lines, summary_prompt):
    state = _load_single_pass_full_polish_progress_state()
    line_count = len(src_lines)
    clean_draft_lines = [_sanitize_single_line_text(line) for line in draft_lines]
    source_hash = _hash_lines(src_lines)

    if not state:
        _persist_single_pass_full_polish_progress_state(
            src_lines=src_lines,
            draft_lines=clean_draft_lines,
            polished_prefix_lines=[],
            summary_prompt=summary_prompt,
            status="in_progress",
            final_line_marker_seen=False,
        )
        return {
            "draft_lines": clean_draft_lines,
            "polished_lines": clean_draft_lines.copy(),
            "next_line": 1,
            "resumed": False,
            "completed": False,
        }

    if state.get("version") != POLISH_PROGRESS_STATE_VERSION:
        console.print("[yellow]Single-pass polish progress version mismatch; starting a new session.[/yellow]")
    elif state.get("line_count") != line_count:
        console.print("[yellow]Single-pass polish progress line count mismatch; starting a new session.[/yellow]")
    elif state.get("source_hash") != source_hash:
        console.print("[yellow]Single-pass polish progress source mismatch; starting a new session.[/yellow]")
    else:
        saved_draft_lines = state.get("base_draft_lines") or []
        saved_prefix_lines = state.get("polished_prefix_lines") or []
        saved_prefix_lines = [
            _sanitize_single_line_text(line) for line in saved_prefix_lines[:line_count]
        ]
        saved_draft_lines = [
            _sanitize_single_line_text(line) for line in saved_draft_lines[:line_count]
        ]
        if len(saved_draft_lines) == line_count:
            committed_count = len(saved_prefix_lines)
            polished_lines = saved_draft_lines.copy()
            polished_lines[:committed_count] = saved_prefix_lines
            final_line_marker_seen = bool(state.get("final_line_marker_seen"))
            completion_reason = state.get("completion_reason")

            if _hash_lines(clean_draft_lines) != state.get("draft_hash"):
                console.print(
                    "[yellow]Detected existing single-pass polish progress. "
                    "Resuming from the saved draft instead of the newly generated draft.[/yellow]"
                )

            next_line = committed_count + 1
            if final_line_marker_seen and next_line <= line_count:
                polished_lines = _force_complete_single_pass_full_polish_tail(
                    src_lines=src_lines,
                    draft_lines=saved_draft_lines,
                    polished_lines=polished_lines,
                    summary_prompt=summary_prompt,
                    start_line=next_line,
                )
                return {
                    "draft_lines": saved_draft_lines,
                    "polished_lines": polished_lines,
                    "next_line": line_count + 1,
                    "resumed": True,
                    "completed": True,
                    "completion_reason": "forced_tail_placeholder",
                }

            if next_line > line_count:
                if completion_reason == "forced_tail_placeholder":
                    console.print(
                        "[yellow]Single-pass full polish progress already contains forced tail placeholders. "
                        "Reusing saved result.[/yellow]"
                    )
                return {
                    "draft_lines": saved_draft_lines,
                    "polished_lines": polished_lines,
                    "next_line": next_line,
                    "resumed": True,
                    "completed": True,
                    "completion_reason": completion_reason,
                }

            console.print(
                f"[cyan]Resuming single-pass full polish with {committed_count}/{line_count} confirmed lines. "
                f"Continuing from line {next_line} "
                f"using {_4_2_SINGLE_PASS_FULL_POLISH_PREVIEW}.[/cyan]"
            )
            return {
                "draft_lines": saved_draft_lines,
                "polished_lines": polished_lines,
                "next_line": next_line,
                "resumed": True,
                "completed": False,
                "completion_reason": completion_reason,
            }

        console.print("[yellow]Single-pass polish progress file is incomplete; starting a new session.[/yellow]")

    _persist_single_pass_full_polish_progress_state(
        src_lines=src_lines,
        draft_lines=clean_draft_lines,
        polished_prefix_lines=[],
        summary_prompt=summary_prompt,
        status="in_progress",
        final_line_marker_seen=False,
    )
    return {
        "draft_lines": clean_draft_lines,
        "polished_lines": clean_draft_lines.copy(),
        "next_line": 1,
        "resumed": False,
        "completed": False,
        "completion_reason": None,
    }

def _resolve_single_pass_full_polish_request_settings(api_settings):
    global_api_key = _load_key_or_default("api.key", "")
    global_model = _load_key_or_default("api.model", "")
    global_base_url = _load_key_or_default("api.base_url", "")
    global_timeout = _load_key_or_default("api.request_timeout_sec", 300)
    global_retries = _load_key_or_default("api.request_retries", 5)
    global_retry_delay = _load_key_or_default("api.request_retry_delay_sec", 1)

    api_key = _pick_setting(api_settings, "key", global_api_key)
    if not api_key:
        raise ValueError("API key is not set")

    model = _pick_setting(api_settings, "model", global_model)
    if not model:
        raise ValueError("Model is not set")

    base_url = _pick_setting(api_settings, "base_url", global_base_url)
    timeout = _to_int(
        _pick_setting(api_settings, "request_timeout_sec", global_timeout),
        300,
        min_value=1,
    )
    retries = _to_int(
        _pick_setting(api_settings, "request_retries", global_retries),
        5,
        min_value=0,
    )
    retry_delay = _to_float(
        _pick_setting(api_settings, "request_retry_delay_sec", global_retry_delay),
        1,
        min_value=0,
    )
    route = "claude-messages" if _is_claude_model(model) else "openai-chat"

    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "timeout": timeout,
        "retries": retries,
        "retry_delay": retry_delay,
        "route": route,
    }

def _stream_openai_text(prompt, request_settings, on_delta):
    client_kwargs = {"api_key": request_settings["api_key"]}
    if request_settings["base_url"]:
        client_kwargs["base_url"] = _normalize_openai_base_url(request_settings["base_url"])

    client = OpenAI(**client_kwargs)
    stream = client.chat.completions.create(
        model=request_settings["model"],
        messages=[{"role": "user", "content": prompt}],
        timeout=request_settings["timeout"],
        stream=True,
    )

    finish_reason = None
    try:
        for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if content:
                on_delta(content)

            chunk_finish_reason = getattr(choice, "finish_reason", None)
            if chunk_finish_reason:
                finish_reason = chunk_finish_reason
    finally:
        close_stream = getattr(stream, "close", None)
        if callable(close_stream):
            close_stream()

    return finish_reason or "stop"

def _stream_claude_text(prompt, request_settings, on_delta):
    if not request_settings["base_url"]:
        raise ValueError("Claude base_url is not set")

    url = _normalize_claude_messages_url(request_settings["base_url"])
    headers = {
        "x-api-key": request_settings["api_key"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": request_settings["model"],
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }

    finish_reason = None
    with requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=request_settings["timeout"],
        stream=True,
    ) as response:
        response.raise_for_status()
        event_type = None
        data_lines = []

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue

            line = raw_line.strip()
            if not line:
                if data_lines:
                    payload_text = "\n".join(data_lines)
                    payload_data = json.loads(payload_text)

                    if event_type == "content_block_delta":
                        delta = payload_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                on_delta(text)
                    elif event_type == "message_delta":
                        finish_reason = (
                            payload_data.get("delta", {}).get("stop_reason") or finish_reason
                        )
                    elif event_type == "error":
                        error_info = payload_data.get("error", payload_data)
                        raise ValueError(f"Claude streaming error: {error_info}")
                    elif event_type == "message_stop":
                        finish_reason = finish_reason or "stop"
                        break

                event_type = None
                data_lines = []
                continue

            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())

    return finish_reason or "stop"

def _parse_polish_stream_lines(raw_content, start_line, end_line, allow_last_without_newline=False):
    if not raw_content:
        return []

    normalized = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    physical_lines = normalized.splitlines(keepends=True)
    parsed_lines = {}

    for i, line in enumerate(physical_lines):
        is_last_line = i == len(physical_lines) - 1
        if is_last_line and not line.endswith("\n") and not allow_last_without_newline:
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue

        match = POLISH_LINE_PATTERN.match(stripped)
        if not match:
            continue

        line_no = int(match.group(1))
        polished_text = match.group(2).strip()
        if line_no < start_line or line_no > end_line or not polished_text:
            continue
        if line_no not in parsed_lines:
            parsed_lines[line_no] = polished_text

    contiguous_lines = []
    current_line = start_line
    while current_line in parsed_lines:
        contiguous_lines.append((current_line, parsed_lines[current_line]))
        current_line += 1

    return contiguous_lines

def _commit_polish_stream_progress(
    raw_content,
    polished_lines,
    draft_lines,
    summary_prompt,
    src_lines,
    start_line,
    end_line,
    committed_count,
    final_line_marker_seen,
    allow_last_without_newline=False,
):
    parsed_lines = _parse_polish_stream_lines(
        raw_content,
        start_line,
        end_line,
        allow_last_without_newline=allow_last_without_newline,
    )
    new_lines = parsed_lines[committed_count:]
    for line_no, polished_text in new_lines:
        polished_lines[line_no - 1] = polished_text
        console.print(f"[green][{line_no}][/green] {polished_text}")
        _persist_single_pass_full_polish_progress_state(
            src_lines=src_lines,
            draft_lines=draft_lines,
            polished_prefix_lines=polished_lines[:line_no],
            summary_prompt=summary_prompt,
            status="completed" if line_no == len(src_lines) else "in_progress",
            final_line_marker_seen=final_line_marker_seen,
            completion_reason=None,
        )
    return len(new_lines), len(parsed_lines)

def _log_single_pass_full_polish_round(
    request_settings,
    prompt,
    raw_content,
    round_index,
    start_line,
    committed_count,
    finish_reason=None,
    error=None,
):
    message = (
        f"round={round_index} start_line={start_line} committed={committed_count} "
        f"finish_reason={finish_reason or 'unknown'}"
    )
    if error:
        message += f" error={error}"

    _save_cache(
        request_settings["model"],
        prompt,
        raw_content,
        "text",
        raw_content,
        message=message,
        log_title="single_pass_full_polish",
    )

def _run_single_pass_full_polish_round(
    prompt,
    src_lines,
    draft_lines,
    summary_prompt,
    polished_lines,
    start_line,
    line_count,
    request_settings,
    round_index,
):
    raw_content = ""
    committed_count = 0
    last_reported_count = 0
    last_stream_snapshot_at = 0.0
    finish_reason = None
    error = None
    final_line_marker_seen = False

    def persist_stream_snapshot(force=False):
        nonlocal last_stream_snapshot_at
        now = time.time()
        if not force and now - last_stream_snapshot_at < POLISH_STREAM_SNAPSHOT_INTERVAL_SEC:
            return
        _persist_single_pass_full_polish_stream_snapshot(
            start_line=start_line,
            line_count=line_count,
            committed_count=committed_count,
            final_line_marker_seen=final_line_marker_seen,
            raw_content=raw_content,
        )
        last_stream_snapshot_at = now

    def report_progress(force=False):
        nonlocal last_reported_count
        if committed_count <= last_reported_count:
            return
        if not force and committed_count - last_reported_count < POLISH_PROGRESS_REPORT_STEP:
            return

        absolute_done = start_line + committed_count - 1
        console.print(
            f"[cyan]Full-text polish progress: {absolute_done}/{line_count} lines committed.[/cyan]"
        )
        last_reported_count = committed_count

    def on_delta(text):
        nonlocal raw_content, committed_count, final_line_marker_seen
        raw_content += text
        if not final_line_marker_seen and _has_seen_target_line_marker(raw_content, line_count):
            final_line_marker_seen = True
            _persist_single_pass_full_polish_progress_state(
                src_lines=src_lines,
                draft_lines=draft_lines,
                polished_prefix_lines=polished_lines[: start_line + committed_count - 1],
                summary_prompt=summary_prompt,
                status="in_progress",
                final_line_marker_seen=True,
                completion_reason=None,
            )
            console.print(
                f"[yellow]Detected final line marker [{line_count}] in streaming output. "
                "If the stream dies now, the remaining tail will be force-completed on resume.[/yellow]"
            )
            persist_stream_snapshot(force=True)
        if "\n" not in text:
            persist_stream_snapshot(force=False)
            return

        _, parsed_count = _commit_polish_stream_progress(
            raw_content,
            polished_lines,
            draft_lines,
            summary_prompt,
            src_lines,
            start_line,
            line_count,
            committed_count,
            final_line_marker_seen,
            allow_last_without_newline=False,
        )
        committed_count = parsed_count
        report_progress(force=False)
        persist_stream_snapshot(force=True)

    try:
        if request_settings["route"] == "claude-messages":
            finish_reason = _stream_claude_text(prompt, request_settings, on_delta)
        else:
            finish_reason = _stream_openai_text(prompt, request_settings, on_delta)
    except Exception as exc:
        error = exc
    finally:
        _, parsed_count = _commit_polish_stream_progress(
            raw_content,
            polished_lines,
            draft_lines,
            summary_prompt,
            src_lines,
            start_line,
            line_count,
            committed_count,
            final_line_marker_seen,
            allow_last_without_newline=error is None,
        )
        committed_count = parsed_count
        if final_line_marker_seen and start_line + committed_count - 1 < line_count:
            first_unfinished_line = start_line + committed_count
            polished_lines = _force_complete_single_pass_full_polish_tail(
                src_lines=src_lines,
                draft_lines=draft_lines,
                polished_lines=polished_lines,
                summary_prompt=summary_prompt,
                start_line=first_unfinished_line,
            )
            committed_count = line_count - start_line + 1
        persist_stream_snapshot(force=True)
        report_progress(force=True)
        _log_single_pass_full_polish_round(
            request_settings=request_settings,
            prompt=prompt,
            raw_content=raw_content,
            round_index=round_index,
            start_line=start_line,
            committed_count=committed_count,
            finish_reason=finish_reason,
            error=error,
        )

    return {
        "committed": committed_count,
        "finish_reason": finish_reason,
        "error": error,
    }

def polish_single_pass_full_text(src_lines, draft_lines, summary_prompt=None):
    if len(src_lines) != len(draft_lines):
        raise ValueError("Full polish input mismatch: source and translation line counts are different.")

    line_count = len(draft_lines)
    api_settings = _load_single_pass_full_polish_api_settings()
    request_settings = _resolve_single_pass_full_polish_request_settings(api_settings)
    resume_state = _restore_single_pass_full_polish_progress(
        src_lines=src_lines,
        draft_lines=draft_lines,
        summary_prompt=summary_prompt,
    )
    draft_lines = resume_state["draft_lines"]
    polished_lines = resume_state["polished_lines"]
    next_line = resume_state["next_line"]
    no_progress_attempts = 0
    round_index = 0

    if resume_state["completed"]:
        console.print("[green]Single-pass full polish already completed in progress file. Reusing saved result.[/green]")
        return polished_lines

    while next_line <= line_count:
        round_index += 1
        prompt = get_prompt_single_pass_full_polish(
            source_lines=src_lines,
            draft_lines=draft_lines,
            summary_prompt=summary_prompt,
            start_line=next_line,
            polished_prefix_lines=polished_lines[: next_line - 1],
        )
        console.print(
            f"[cyan]Full-text polish round {round_index}: streaming lines {next_line}-{line_count}...[/cyan]"
        )
        round_result = _run_single_pass_full_polish_round(
            prompt=prompt,
            src_lines=src_lines,
            draft_lines=draft_lines,
            summary_prompt=summary_prompt,
            polished_lines=polished_lines,
            start_line=next_line,
            line_count=line_count,
            request_settings=request_settings,
            round_index=round_index,
        )

        committed = round_result["committed"]
        if committed > 0:
            next_line += committed
            no_progress_attempts = 0
            if next_line <= line_count:
                console.print(
                    "[yellow]Full-text polish stream ended before completion; "
                    f"resuming from line {next_line}. "
                    f"finish_reason={round_result['finish_reason'] or 'unknown'}[/yellow]"
                )
            continue

        no_progress_attempts += 1
        max_attempts = request_settings["retries"] + 1
        if no_progress_attempts >= max_attempts:
            last_reason = round_result["error"] or (
                f"finish_reason={round_result['finish_reason'] or 'unknown'}"
            )
            raise ValueError(
                f"Full polish made no progress from line {next_line} after {max_attempts} attempts. "
                f"Last reason: {last_reason}"
            )

        console.print(
            "[yellow]Full-text polish made no progress; "
            f"retrying from line {next_line} "
            f"({no_progress_attempts + 1}/{max_attempts})...[/yellow]"
        )
        if request_settings["retry_delay"] > 0:
            time.sleep(request_settings["retry_delay"])

    for i, polished in enumerate(polished_lines, 1):
        if not polished and str(src_lines[i - 1]).strip():
            raise ValueError(f"Full polish output mismatch: empty polished line at {i}.")

    return polished_lines

# 🚀 翻译所有块的主函数
@check_file_exists(_4_2_TRANSLATION)
def translate_all():
    console.print("[bold green]Start Translating All...[/bold green]")
    chunks = split_chunks_by_chars(chunk_size=600, max_i=10)
    with open(_4_1_TERMINOLOGY, 'r', encoding='utf-8') as file:
        theme_prompt = json.load(file).get('theme')

    # 🔄 使用并发执行翻译
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("[cyan]Translating chunks...", total=len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            futures = []
            for i, chunk in enumerate(chunks):
                future = executor.submit(translate_chunk, chunk, chunks, theme_prompt, i)
                futures.append(future)
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(task, advance=1)

    results.sort(key=lambda x: x[0])  # 按原始顺序排序结果
    
    # 💾 保存结果到列表和Excel文件
    src_text, trans_text = [], []
    for i, chunk in enumerate(chunks):
        chunk_lines = chunk.split('\n')
        src_text.extend(chunk_lines)
        
        # 计算当前块与翻译结果的相似度
        chunk_text = ''.join(chunk_lines).lower()
        matching_results = [(r, similar(''.join(r[1].split('\n')).lower(), chunk_text)) 
                          for r in results]
        best_match = max(matching_results, key=lambda x: x[1])
        
        # 检查相似度并处理异常
        if best_match[1] < 0.9:
            console.print(f"[yellow]Warning: No matching translation found for chunk {i}[/yellow]")
            raise ValueError(f"Translation matching failed (chunk {i})")
        elif best_match[1] < 1.0:
            console.print(f"[yellow]Warning: Similar match found (chunk {i}, similarity: {best_match[1]:.3f})[/yellow]")
            
        trans_text.extend(best_match[0][2].split('\n'))

    # single-pass mode: add one full-text polish pass after chunk translation
    if not load_key("reflect_translate"):
        console.print("[cyan]Single-pass mode detected: running full-text polish...[/cyan]")
        try:
            trans_text = polish_single_pass_full_text(src_text, trans_text, theme_prompt)
            console.print("[green]✅ Full-text polish completed.[/green]")
        except Exception as e:
            console.print(
                "[yellow]Warning: Full-text polish failed; fallback to original single-pass result. "
                f"Reason: {e}[/yellow]"
            )
    
    # 裁剪过长的翻译文本
    df_text = pd.read_excel(_2_CLEANED_CHUNKS)
    df_text['text'] = df_text['text'].str.strip('"').str.strip()
    df_translate = pd.DataFrame({'Source': src_text, 'Translation': trans_text})
    subtitle_output_configs = [('trans_subs_for_audio.srt', ['Translation'])]
    df_time = align_timestamp(df_text, df_translate, subtitle_output_configs, output_dir=None, for_display=False)
    console.print(df_time)
    # 对 df_time['Translation'] 应用 check_len_then_trim，仅当 duration > MIN_TRIM_DURATION 时
    df_time['Translation'] = df_time.apply(lambda x: check_len_then_trim(x['Translation'], x['duration']) if x['duration'] > load_key("min_trim_duration") else x['Translation'], axis=1)
    console.print(df_time)
    
    df_time.to_excel(_4_2_TRANSLATION, index=False)
    _cleanup_single_pass_full_polish_progress()
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
