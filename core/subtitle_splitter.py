import pandas as pd
from typing import List, Tuple
import concurrent.futures
import json
import os

from core.splitter_meaning import split_sentence
from core.prompts import get_align_prompt, get_target_semantic_alignment_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.paths import *
console = Console()

SUBTITLE_SPLIT_DECISIONS = "output/log/subtitle_split_decisions.xlsx"
TARGET_ALIGNMENT_BATCH_SIZE = 20
TARGET_BOUNDARY_PUNCTUATION = set("，。！？；：,.!?;:…")
TARGET_BOUNDARY_CLOSERS = set("」』】）》〉〕”’)]}")

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)


def _is_han_character(char: str) -> bool:
    if not char:
        return False
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def detect_target_boundary_candidates(target: str) -> List[str]:
    """把目标字幕按可靠候选边界切成可相邻合并的原子单元。"""
    text = str(target).strip()
    if not text:
        return []

    boundaries = []
    i = 0
    text_length = len(text)

    while i < text_length:
        char = text[i]

        # 中文字符之间的空白可以作为边界；英文词组内部空格会保留在同一单元。
        if char.isspace():
            whitespace_end = i + 1
            while whitespace_end < text_length and text[whitespace_end].isspace():
                whitespace_end += 1
            if (
                i > 0
                and whitespace_end < text_length
                and _is_han_character(text[i - 1])
                and _is_han_character(text[whitespace_end])
            ):
                boundaries.append((i, whitespace_end))
            i = whitespace_end
            continue

        if char in TARGET_BOUNDARY_PUNCTUATION:
            punctuation_start = i
            punctuation_end = i + 1
            while (
                punctuation_end < text_length
                and text[punctuation_end] in TARGET_BOUNDARY_PUNCTUATION
            ):
                punctuation_end += 1

            # 小数、版本号、时间等内部 ASCII 标点保留在当前单元。
            punctuation_text = text[punctuation_start:punctuation_end]
            if (
                len(punctuation_text) == 1
                and punctuation_text in ".,:"
                and punctuation_start > 0
                and punctuation_end < text_length
                and text[punctuation_start - 1].isascii()
                and text[punctuation_start - 1].isalnum()
                and text[punctuation_end].isascii()
                and text[punctuation_end].isalnum()
            ):
                i = punctuation_end
                continue

            while (
                punctuation_end < text_length
                and text[punctuation_end] in TARGET_BOUNDARY_CLOSERS
            ):
                punctuation_end += 1

            next_content = punctuation_end
            while next_content < text_length and text[next_content].isspace():
                next_content += 1

            if (
                len(punctuation_text) == 1
                and punctuation_text in ".,:，。："
                and punctuation_start > 0
                and next_content < text_length
                and text[punctuation_start - 1].isdigit()
                and text[next_content].isdigit()
            ):
                i = next_content
                continue

            # 英文术语、型号、时间和 URL 内部的 ASCII 标点保留在当前单元。
            ascii_punctuation = all(mark in ".,!?;:" for mark in punctuation_text)
            if (
                ascii_punctuation
                and punctuation_start > 0
                and next_content < text_length
                and text[punctuation_start - 1].isascii()
                and text[punctuation_start - 1].isalnum()
                and text[next_content].isascii()
                and (
                    text[next_content].isalnum()
                    or text[next_content] in "/\\@#%&=_+-"
                )
            ):
                i = next_content
                continue

            if text[:punctuation_start].strip() and text[next_content:].strip():
                boundaries.append((punctuation_end, next_content))
            i = next_content
            continue

        i += 1

    if not boundaries:
        return []

    units = []
    cursor = 0
    for left_end, right_start in boundaries:
        if left_end < cursor:
            continue
        unit = text[cursor:left_end].strip()
        if unit:
            units.append(unit)
        cursor = max(cursor, right_start)

    tail = text[cursor:].strip()
    if tail:
        units.append(tail)

    return units if len(units) > 1 else []


def _locate_candidate_unit_spans(candidate_units: List[str], original_target: str):
    spans = []
    cursor = 0
    for unit in candidate_units:
        start = original_target.find(unit, cursor)
        if start < 0:
            return []
        end = start + len(unit)
        spans.append((start, end))
        cursor = end
    return spans


def _match_adjacent_candidate_units(
    target_parts: List[str],
    candidate_units: List[str],
    original_target: str,
) -> bool:
    """校验每段由相邻单元组成，并按顺序完整覆盖；连续时间段可重复同一组单元。"""
    if not target_parts or not candidate_units:
        return False

    original_target = str(original_target).strip()
    unit_spans = _locate_candidate_unit_spans(candidate_units, original_target)
    matching_ranges = []
    for target_part in target_parts:
        part_ranges = []
        for start in range(len(candidate_units)):
            combined = ""
            for end in range(start + 1, len(candidate_units) + 1):
                combined += candidate_units[end - 1]
                allowed_texts = {combined}
                if unit_spans:
                    original_slice = original_target[
                        unit_spans[start][0]:unit_spans[end - 1][1]
                    ].strip()
                    allowed_texts.add(original_slice)
                if target_part in allowed_texts:
                    part_ranges.append((start, end))
                if len(combined) > len(target_part):
                    break
        if not part_ranges:
            return False
        matching_ranges.append(part_ranges)

    states = {(start, end) for start, end in matching_ranges[0] if start == 0}
    for part_ranges in matching_ranges[1:]:
        next_states = set()
        for start, end in part_ranges:
            for previous_start, previous_end in states:
                repeats_same_units = (
                    start == previous_start and end == previous_end
                )
                continues_from_previous = start == previous_end
                if repeats_same_units or continues_from_previous:
                    next_states.add((start, end))
                    break
        states = next_states
        if not states:
            return False

    return any(end == len(candidate_units) for _, end in states)


def _validate_target_alignment_response(response_data, expected_items):
    if not isinstance(response_data, dict) or not isinstance(response_data.get("items"), list):
        return False, "Missing required list: `items`", {}
    if set(response_data) != {"items"}:
        return False, "Unexpected fields in target alignment response", {}

    response_items = response_data["items"]
    if len(response_items) != len(expected_items):
        return False, "Target alignment item count mismatch", {}

    response_ids = [item.get("id") if isinstance(item, dict) else None for item in response_items]
    expected_ids = [item["id"] for item in expected_items]
    if response_ids != expected_ids:
        return False, "Target alignment item order or ids mismatch", {}

    expected_by_id = {item["id"]: item for item in expected_items}
    parts_by_id = {}

    for item in response_items:
        if not isinstance(item, dict):
            return False, "Target alignment item must be an object", {}
        if set(item) != {"id", "target_parts"}:
            return False, "Unexpected fields in target alignment item", {}

        item_id = item.get("id")
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            return False, "Target alignment id must be an integer", {}
        if item_id not in expected_by_id or item_id in parts_by_id:
            return False, f"Unexpected or duplicate target alignment id: {item_id}", {}

        target_parts = item.get("target_parts")
        expected = expected_by_id[item_id]
        if not isinstance(target_parts, list):
            return False, f"target_parts must be a list for id {item_id}", {}
        if len(target_parts) != len(expected["source_parts"]):
            return False, f"target_parts count mismatch for id {item_id}", {}
        if any(not isinstance(part, str) or not part.strip() for part in target_parts):
            return False, f"target_parts contains an empty or non-string value for id {item_id}", {}

        clean_parts = [part.strip() for part in target_parts]
        if not _match_adjacent_candidate_units(
            clean_parts,
            expected["target_boundary_candidates"],
            expected["target"],
        ):
            return False, f"target_parts changed candidate order or content for id {item_id}", {}
        parts_by_id[item_id] = clean_parts

    if set(parts_by_id) != set(expected_by_id):
        return False, "Target alignment ids mismatch", {}

    return True, "Target semantic alignment completed", parts_by_id

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('\n')
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def _align_target_semantics_batch(batch_items):
    def valid_target_alignment(response_data):
        valid, message, _ = _validate_target_alignment_response(response_data, batch_items)
        return {
            "status": "success" if valid else "error",
            "message": message,
        }

    try:
        prompt = get_target_semantic_alignment_prompt(batch_items)
        response_data = ask_gpt(
            prompt,
            resp_type='json',
            valid_def=valid_target_alignment,
            log_title='align_target_semantics',
        )
        valid, message, parts_by_id = _validate_target_alignment_response(
            response_data,
            batch_items,
        )
        if not valid:
            return {}, message
        return parts_by_id, ""
    except Exception as error:
        return {}, f"{type(error).__name__}: {error}"


def _serialize_decision_parts(parts) -> str:
    return json.dumps(parts, ensure_ascii=False)


def split_align_subs(
    src_lines: List[str],
    tr_lines: List[str],
    attempt: int = 1,
    decision_records=None,
):
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]

    result_src_lines = list(src_lines)
    result_tr_lines = list(tr_lines)
    to_split = []
    round_records = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src, tr = str(src), str(tr)
        src_over = len(src) > MAX_SUB_LENGTH
        target_length = calc_len(tr)
        adjusted_target_length = target_length * TARGET_SUB_MULTIPLIER
        tr_over = adjusted_target_length > MAX_SUB_LENGTH
        mode = "keep"
        if src_over or tr_over:
            # 翻译超长时继续使用原有的拆分与对齐流程。
            mode = "split+align_translation" if tr_over else "split_source_only"

        round_records.append(
            {
                "attempt": attempt,
                "line_index": i,
                "source": src,
                "target": tr,
                "source_length": len(src),
                "target_weighted_length": target_length,
                "target_adjusted_length": adjusted_target_length,
                "max_length": MAX_SUB_LENGTH,
                "target_multiplier": TARGET_SUB_MULTIPLIER,
                "source_over": src_over,
                "target_over": tr_over,
                "mode": mode,
                "target_boundary_candidates": _serialize_decision_parts([]),
                "decision": "keep",
                "fallback_reason": "",
                "source_parts": _serialize_decision_parts([src]),
                "target_parts": _serialize_decision_parts([tr]),
            }
        )

        if src_over or tr_over:
            to_split.append(
                {
                    "index": i,
                    "mode": mode,
                    "src_over": src_over,
                    "tr_over": tr_over,
                    "source": src,
                    "target": tr,
                }
            )
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src)
            table.add_row("Target Line", tr)
            table.add_row("Trigger", f"src_over={src_over}, tr_over={tr_over}, mode={mode}")
            console.print(table)
    
    @except_handler("Error in split_align_subs")
    def process(split_item):
        i = split_item["index"]
        mode = split_item["mode"]
        source = split_item["source"]
        target = split_item["target"]
        split_src = split_sentence(source, num_parts=2).strip()

        if mode == "split+align_translation":
            src_parts, tr_parts, _ = align_subs(source, target, split_src)
            return {
                "index": i,
                "mode": mode,
                "source_parts": src_parts,
                "target_parts": tr_parts,
                "target_boundary_candidates": [],
                "decision": "split+align_translation",
                "fallback_reason": "",
            }
        elif mode == "split_source_only":
            src_parts = [part.strip() for part in split_src.split("\n") if str(part).strip()]
            if not src_parts:
                src_parts = [source.strip()]
            return {
                "index": i,
                "mode": mode,
                "source_parts": src_parts,
                "target": target.strip(),
                "target_boundary_candidates": detect_target_boundary_candidates(target),
            }
        else:
            raise ValueError(f"Unsupported split mode: {mode}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        processed_items = list(executor.map(process, to_split))

    semantic_items = [
        {
            "id": item["index"],
            "source": str(src_lines[item["index"]]),
            "source_parts": item["source_parts"],
            "target": item["target"],
            "target_boundary_candidates": item["target_boundary_candidates"],
        }
        for item in processed_items
        if (
            item["mode"] == "split_source_only"
            and len(item["source_parts"]) > 1
            and item["target_boundary_candidates"]
        )
    ]
    semantic_batches = [
        semantic_items[start:start + TARGET_ALIGNMENT_BATCH_SIZE]
        for start in range(0, len(semantic_items), TARGET_ALIGNMENT_BATCH_SIZE)
    ]

    semantic_parts_by_id = {}
    semantic_errors_by_id = {}
    if semantic_batches:
        with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
            batch_results = list(executor.map(_align_target_semantics_batch, semantic_batches))

        for batch_items, (parts_by_id, error_message) in zip(semantic_batches, batch_results):
            if error_message:
                for batch_item in batch_items:
                    semantic_errors_by_id[batch_item["id"]] = error_message
                console.print(
                    "[yellow]Target semantic alignment failed for batch; "
                    "using the complete target for each source part.[/yellow]"
                )
            else:
                semantic_parts_by_id.update(parts_by_id)

    for item in processed_items:
        i = item["index"]
        source_parts = item["source_parts"]

        if item["mode"] == "split_source_only":
            target = item["target"]
            candidates = item["target_boundary_candidates"]
            if len(source_parts) <= 1:
                target_parts = [target]
                decision = "repeat_target_without_source_split"
                fallback_reason = "source split produced one part"
            elif i in semantic_parts_by_id:
                target_parts = semantic_parts_by_id[i]
                decision = "align_target_semantics"
                fallback_reason = ""
            else:
                target_parts = [target] * len(source_parts)
                if candidates:
                    decision = "repeat_target_after_alignment_fallback"
                    fallback_reason = semantic_errors_by_id.get(i, "missing validated batch item")
                else:
                    decision = "repeat_target_without_candidates"
                    fallback_reason = "no target boundary candidates"
        else:
            target_parts = item["target_parts"]
            candidates = item["target_boundary_candidates"]
            decision = item["decision"]
            fallback_reason = item["fallback_reason"]

        result_src_lines[i] = source_parts
        result_tr_lines[i] = target_parts

        record = round_records[i]
        record["target_boundary_candidates"] = _serialize_decision_parts(candidates)
        record["decision"] = decision
        record["fallback_reason"] = fallback_reason
        record["source_parts"] = _serialize_decision_parts(source_parts)
        record["target_parts"] = _serialize_decision_parts(target_parts)

    if decision_records is not None:
        decision_records.extend(round_records)

    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in result_src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in result_tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    
    return src_lines, tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")
    
    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()
    
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    decision_records = []

    for attempt in range(3):  # 多次切割
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        split_src, split_trans = split_align_subs(
            src.copy(),
            trans,
            attempt=attempt + 1,
            decision_records=decision_records,
        )
        
        # 检查是否所有字幕都符合长度要求
        if all(len(src) <= MAX_SUB_LENGTH for src in split_src) and \
           all(calc_len(tr) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for tr in split_trans):
            break
        
        # 更新源数据继续下一轮分割
        src, trans = split_src, split_trans

    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    os.makedirs(os.path.dirname(SUBTITLE_SPLIT_DECISIONS), exist_ok=True)
    pd.DataFrame(decision_records).to_excel(SUBTITLE_SPLIT_DECISIONS, index=False)

if __name__ == '__main__':
    split_for_sub_main()
