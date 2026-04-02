"""
MFA 强制对齐模块（实验性功能）

使用 Montreal Forced Aligner 优化 stable-ts 产出的词级时间戳。
核心思路：保留 stable-ts 识别的文本，仅用 MFA 重新对齐时间戳。

工作流程：
1. 从 DataFrame 提取词序列，生成 MFA 输入文件
2. 调用 MFA CLI 进行声学对齐
3. 解析 MFA 输出的 TextGrid，提取精确时间戳
4. 用新时间戳更新 DataFrame（保留原始文本）
"""

import os
import re
import shutil
import tempfile
import subprocess
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from pydub import AudioSegment
from rich import print as rprint
from core.utils import load_key

MAX_MFA_SKIP_WORDS = 3
MAX_LATE_START_SHIFT_SEC = 1.0
LATE_SHIFT_RETRY_PAD_BEFORE_SEC = 1.0
LATE_SHIFT_RETRY_PAD_AFTER_SEC = 1.0
OVERLAP_RETRY_PAD_SEC = 2.0
OVERLAP_RETRY_MAX_PASSES = 2

def _load_mfa_config(key: str, default):
    try:
        value = load_key(key)
    except Exception:
        return default

    if value in (None, ""):
        return default
    return value


def _normalize_word(text: str) -> str:
    return str(text).strip('"').strip("'").strip().lower()


def _words_match(stable_word: str, mfa_word: str) -> bool:
    if not stable_word or not mfa_word:
        return False
    return stable_word == mfa_word or stable_word in mfa_word or mfa_word in stable_word

def check_mfa_available() -> bool:
    """
    检查 MFA 是否可用
    
    Returns:
        是否可用
    """
    try:
        result = subprocess.run(
            ['mfa', 'version'],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def prepare_mfa_input(df: pd.DataFrame, audio_file: str, work_dir: str) -> Tuple[str, str]:
    """
    准备 MFA 输入文件
    
    Args:
        df: stable-ts 输出的 DataFrame，包含 text, start, end 列
        audio_file: 音频文件路径
        work_dir: 工作目录
    
    Returns:
        (音频文件路径, 文本文件路径)
    """
    # 创建输入目录
    input_dir = os.path.join(work_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    
    # 复制音频文件到输入目录（MFA 需要音频和文本在同一目录）
    audio_ext = os.path.splitext(audio_file)[1]
    audio_dest = os.path.join(input_dir, f'audio{audio_ext}')
    shutil.copy2(audio_file, audio_dest)
    
    # 生成文本文件（所有词连成一个文本）
    # 清理文本中的引号（stable-ts 输出的 text 可能带引号）
    words = []
    for text in df['text'].tolist():
        # 去除引号
        clean_text = str(text).strip('"').strip("'").strip()
        if clean_text:
            words.append(clean_text)
    
    transcript = ' '.join(words)
    txt_path = os.path.join(input_dir, 'audio.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    rprint(f"[cyan]📝 MFA 输入准备完成: {len(words)} 个词[/cyan]")
    return audio_dest, txt_path

def run_mfa_alignment(
    input_dir: str, 
    output_dir: str, 
    acoustic_model: str,
    dictionary: str
) -> bool:
    """
    运行 MFA 对齐
    
    Args:
        input_dir: 包含音频和文本的输入目录
        output_dir: TextGrid 输出目录
        acoustic_model: 声学模型名称
        dictionary: 发音词典名称
    
    Returns:
        是否成功
    """
    rprint(f"[cyan]🎯 运行 MFA 对齐 (模型: {acoustic_model}, 词典: {dictionary})...[/cyan]")
    
    # 直接调用 mfa 命令
    cmd = [
        'mfa', 'align',
        input_dir,
        dictionary,
        acoustic_model,
        output_dir,
        '--clean',  # 清理临时文件
        '--single_speaker',  # 单说话人模式，更快
        '--quiet'  # 减少输出
    ]
    
    rprint(f"[dim]   命令: mfa align ... {acoustic_model}[/dim]")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        rprint(f"[yellow]⚠️ MFA 对齐警告: {result.stderr[:300] if result.stderr else 'unknown'}[/yellow]")
        # 检查输出文件是否生成（有时 MFA 返回非零但仍有输出）
        textgrid_files = [f for f in os.listdir(output_dir) if f.endswith('.TextGrid')] if os.path.exists(output_dir) else []
        if not textgrid_files:
            return False
    
    return True

def parse_textgrid(textgrid_path: str) -> List[Tuple[str, float, float]]:
    """
    解析 TextGrid 文件，提取词级时间戳
    
    Args:
        textgrid_path: TextGrid 文件路径
    
    Returns:
        [(word, start, end), ...]
    """
    words = []
    
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 words 层（MFA 输出的词层通常叫 "words"）
    # TextGrid 格式解析
    in_words_tier = False
    intervals_section = False
    current_interval = {}
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 查找 words 层
        if 'name = "words"' in line:
            in_words_tier = True
        
        # 在 words 层中查找 intervals
        if in_words_tier:
            if 'intervals [' in line:
                intervals_section = True
                current_interval = {}
            elif intervals_section:
                if 'xmin = ' in line:
                    match = re.search(r'xmin = ([\d.]+)', line)
                    if match:
                        current_interval['start'] = float(match.group(1))
                elif 'xmax = ' in line:
                    match = re.search(r'xmax = ([\d.]+)', line)
                    if match:
                        current_interval['end'] = float(match.group(1))
                elif 'text = ' in line:
                    match = re.search(r'text = "([^"]*)"', line)
                    if match:
                        text = match.group(1).strip()
                        if text and 'start' in current_interval and 'end' in current_interval:
                            words.append((text, current_interval['start'], current_interval['end']))
                        current_interval = {}
            
            # 如果遇到新的 tier，停止处理
            if 'class = "IntervalTier"' in line and in_words_tier and len(words) > 0:
                break
        
        i += 1
    
    return words


def _build_mfa_matches(
    df: pd.DataFrame,
    mfa_words: List[Tuple[str, float, float]],
) -> Dict[int, Dict[str, float]]:
    working = df.copy()
    working["clean_text"] = working["text"].apply(_normalize_word)
    mfa_lower = [(_normalize_word(word), float(start), float(end)) for word, start, end in mfa_words]

    matches: Dict[int, Dict[str, float]] = {}
    mfa_idx = 0

    for row_idx, row in working.iterrows():
        if mfa_idx >= len(mfa_lower):
            break

        stable_word = row["clean_text"]
        if not stable_word:
            continue

        for skip_count in range(MAX_MFA_SKIP_WORDS + 1):
            candidate_idx = mfa_idx + skip_count
            if candidate_idx >= len(mfa_lower):
                break

            check_word, check_start, check_end = mfa_lower[candidate_idx]
            if _words_match(stable_word, check_word):
                matches[row_idx] = {
                    "start": check_start,
                    "end": check_end,
                    "mfa_idx": candidate_idx,
                }
                mfa_idx = candidate_idx + 1
                break

    return matches


def _apply_mfa_matches(df: pd.DataFrame, matches: Dict[int, Dict[str, float]]) -> int:
    updated_count = 0
    for row_idx, match in matches.items():
        df.at[row_idx, "start"] = float(match["start"])
        df.at[row_idx, "end"] = float(match["end"])
        updated_count += 1
    return updated_count


def _intervals_intersect(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return max(float(start_a), float(start_b)) <= min(float(end_a), float(end_b))


def _select_window_indices(
    df_current: pd.DataFrame,
    df_original: pd.DataFrame,
    window_start: float,
    window_end: float,
) -> List[int]:
    indices = []
    for row_idx in range(len(df_current)):
        current_start = float(df_current.at[row_idx, "start"])
        current_end = float(df_current.at[row_idx, "end"])
        original_start = float(df_original.at[row_idx, "start"])
        original_end = float(df_original.at[row_idx, "end"])

        if _intervals_intersect(current_start, current_end, window_start, window_end) or _intervals_intersect(
            original_start,
            original_end,
            window_start,
            window_end,
        ):
            indices.append(row_idx)
    return indices


def _prepare_mfa_clip_input(
    audio_clip: AudioSegment,
    transcript_words: List[str],
    work_dir: str,
) -> Tuple[str, str]:
    input_dir = os.path.join(work_dir, "input")
    os.makedirs(input_dir, exist_ok=True)

    audio_dest = os.path.join(input_dir, "audio.wav")
    audio_clip.export(audio_dest, format="wav")

    transcript = " ".join(word for word in transcript_words if word)
    txt_path = os.path.join(input_dir, "audio.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return audio_dest, txt_path


def _run_local_mfa_alignment(
    transcript_words: List[str],
    source_audio: AudioSegment,
    clip_start: float,
    clip_end: float,
    acoustic_model: str,
    dictionary: str,
    work_dir: str,
) -> List[Tuple[str, float, float]]:
    if clip_end <= clip_start:
        return []

    clean_words = [str(word).strip('"').strip("'").strip() for word in transcript_words if str(word).strip()]
    if not clean_words:
        return []

    clip_start_ms = max(0, int(round(float(clip_start) * 1000)))
    clip_end_ms = max(clip_start_ms + 1, int(round(float(clip_end) * 1000)))
    audio_clip = source_audio[clip_start_ms:clip_end_ms]
    if len(audio_clip) <= 0:
        return []

    local_dir = tempfile.mkdtemp(prefix="mfa_local_", dir=work_dir)
    try:
        audio_dest, _ = _prepare_mfa_clip_input(audio_clip, clean_words, local_dir)
        input_dir = os.path.dirname(audio_dest)
        output_dir = os.path.join(local_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        success = run_mfa_alignment(input_dir, output_dir, acoustic_model, dictionary)
        if not success:
            return []

        textgrid_files = [f for f in os.listdir(output_dir) if f.endswith(".TextGrid")]
        if not textgrid_files:
            return []

        textgrid_path = os.path.join(output_dir, textgrid_files[0])
        local_words = parse_textgrid(textgrid_path)
        if not local_words:
            return []

        clip_offset = clip_start_ms / 1000.0
        return [(word, start + clip_offset, end + clip_offset) for word, start, end in local_words]
    finally:
        try:
            shutil.rmtree(local_dir)
        except Exception:
            pass


def _realign_indices_with_local_mfa(
    df: pd.DataFrame,
    indices: List[int],
    source_audio: AudioSegment,
    clip_start: float,
    clip_end: float,
    acoustic_model: str,
    dictionary: str,
    work_dir: str,
    reason: str,
    required_indices: Optional[Set[int]] = None,
) -> Tuple[pd.DataFrame, Set[int]]:
    unique_indices = sorted({int(row_idx) for row_idx in indices if 0 <= int(row_idx) < len(df)})
    if not unique_indices:
        return df, set()

    subset = df.loc[unique_indices].copy().reset_index().rename(columns={"index": "original_index"})
    local_mfa_words = _run_local_mfa_alignment(
        transcript_words=subset["text"].tolist(),
        source_audio=source_audio,
        clip_start=clip_start,
        clip_end=clip_end,
        acoustic_model=acoustic_model,
        dictionary=dictionary,
        work_dir=work_dir,
    )
    if not local_mfa_words:
        return df, set()

    match_subset = subset.drop(columns=["original_index"])
    matches = _build_mfa_matches(match_subset, local_mfa_words)
    if not matches:
        return df, set()

    matched_original_indices = {int(subset.at[row_idx, "original_index"]) for row_idx in matches}
    if required_indices and not set(required_indices).issubset(matched_original_indices):
        required_str = ", ".join(str(idx) for idx in sorted(required_indices))
        rprint(f"[yellow]⚠️ 跳过局部 MFA 重对齐（{reason}）：未覆盖必要词索引 {required_str}[/yellow]")
        return df, set()

    for row_idx, match in matches.items():
        original_idx = int(subset.at[row_idx, "original_index"])
        df.at[original_idx, "start"] = float(match["start"])
        df.at[original_idx, "end"] = float(match["end"])

    rprint(
        f"[cyan]🔁 局部 MFA 重对齐完成（{reason}）："
        f"{len(matches)}/{len(unique_indices)} 个词[/cyan]"
    )
    return df, matched_original_indices


def _find_overlap_blocks(df: pd.DataFrame) -> List[Tuple[int, int]]:
    overlap_blocks = []
    block_start = None

    for row_idx in range(len(df) - 1):
        current_end = float(df.at[row_idx, "end"])
        next_start = float(df.at[row_idx + 1, "start"])

        if next_start < current_end:
            if block_start is None:
                block_start = row_idx
        elif block_start is not None:
            overlap_blocks.append((block_start, row_idx))
            block_start = None

    if block_start is not None:
        overlap_blocks.append((block_start, len(df) - 1))

    return overlap_blocks


def _repair_late_start_shifts(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    global_matches: Dict[int, Dict[str, float]],
    source_audio: AudioSegment,
    acoustic_model: str,
    dictionary: str,
    work_dir: str,
) -> Tuple[pd.DataFrame, int]:
    late_shift_indices = []
    for row_idx, match in global_matches.items():
        original_start = float(original_df.at[row_idx, "start"])
        if float(match["start"]) - original_start > MAX_LATE_START_SHIFT_SEC:
            late_shift_indices.append(row_idx)

    if not late_shift_indices:
        return df, 0

    rprint(
        f"[cyan]🧭 检测到 {len(late_shift_indices)} 个词的 MFA 起点相对 stable-ts "
        f"后移超过 {MAX_LATE_START_SHIFT_SEC:.1f}s，尝试局部重对齐...[/cyan]"
    )

    repaired_count = 0
    covered_indices: Set[int] = set()

    for row_idx in late_shift_indices:
        if row_idx in covered_indices:
            continue

        original_start = float(original_df.at[row_idx, "start"])
        original_end = float(original_df.at[row_idx, "end"])
        matched_end = float(global_matches[row_idx]["end"])
        window_start = max(0.0, original_start - LATE_SHIFT_RETRY_PAD_BEFORE_SEC)
        window_end = max(original_end, matched_end) + LATE_SHIFT_RETRY_PAD_AFTER_SEC

        window_indices = _select_window_indices(df, original_df, window_start, window_end)
        if row_idx not in window_indices:
            window_indices.append(row_idx)

        df, matched_indices = _realign_indices_with_local_mfa(
            df=df,
            indices=window_indices,
            source_audio=source_audio,
            clip_start=window_start,
            clip_end=window_end,
            acoustic_model=acoustic_model,
            dictionary=dictionary,
            work_dir=work_dir,
            reason=f"late start word {row_idx}",
            required_indices={row_idx},
        )
        if matched_indices:
            covered_indices.update(matched_indices)
            repaired_count += len(matched_indices)
            continue

        rprint(
            f"[yellow]⚠️ 词索引 {row_idx} 的局部 MFA 重对齐失败，保留全局 MFA 结果[/yellow]"
        )

    return df, repaired_count


def _repair_overlap_blocks(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    source_audio: AudioSegment,
    acoustic_model: str,
    dictionary: str,
    work_dir: str,
) -> Tuple[pd.DataFrame, int]:
    repaired_count = 0

    for pass_index in range(OVERLAP_RETRY_MAX_PASSES):
        overlap_blocks = _find_overlap_blocks(df)
        if not overlap_blocks:
            break

        rprint(
            f"[cyan]🔀 检测到 {len(overlap_blocks)} 个词级重叠块，开始第 "
            f"{pass_index + 1}/{OVERLAP_RETRY_MAX_PASSES} 轮局部 MFA 重对齐...[/cyan]"
        )

        repaired_this_pass = 0
        for block_start, block_end in overlap_blocks:
            block_indices = list(range(block_start, block_end + 1))
            block_start_time = min(float(df.at[row_idx, "start"]) for row_idx in block_indices)
            block_end_time = max(float(df.at[row_idx, "end"]) for row_idx in block_indices)
            window_start = max(0.0, block_start_time - OVERLAP_RETRY_PAD_SEC)
            window_end = block_end_time + OVERLAP_RETRY_PAD_SEC

            window_indices = _select_window_indices(df, original_df, window_start, window_end)
            if not window_indices:
                window_indices = block_indices

            df, matched_indices = _realign_indices_with_local_mfa(
                df=df,
                indices=window_indices,
                source_audio=source_audio,
                clip_start=window_start,
                clip_end=window_end,
                acoustic_model=acoustic_model,
                dictionary=dictionary,
                work_dir=work_dir,
                reason=f"overlap block {block_start}-{block_end}",
                required_indices=set(block_indices),
            )
            if matched_indices:
                repaired_this_pass += len(matched_indices)
                repaired_count += len(matched_indices)

        if repaired_this_pass == 0:
            break

    return df, repaired_count


def update_timestamps(
    df: pd.DataFrame,
    mfa_words: List[Tuple[str, float, float]],
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    用 MFA 时间戳更新 DataFrame
    
    保留 stable-ts 的原始文本，仅更新 start/end 时间戳。
    使用模糊匹配处理可能的词形差异。
    
    Args:
        df: 原始 DataFrame
        mfa_words: MFA 输出的 [(word, start, end), ...]
    
    Returns:
        更新后的 DataFrame
    """
    df = df.copy()
    matches = _build_mfa_matches(df, mfa_words)
    updated_count = _apply_mfa_matches(df, matches)

    rprint(f"[green]✅ MFA 时间戳更新: {updated_count}/{len(df)} 个词[/green]")
    return df, matches

def align_transcription(df: pd.DataFrame, audio_file: str) -> pd.DataFrame:
    """
    MFA 对齐主入口函数
    
    使用 MFA 优化 stable-ts 产出的时间戳。
    如果 MFA 不可用或对齐失败，返回原始 DataFrame。
    
    Args:
        df: stable-ts 输出的 DataFrame
        audio_file: 音频文件路径
    
    Returns:
        优化时间戳后的 DataFrame
    """
    rprint("[cyan]🔧 [实验性] MFA 强制对齐启动...[/cyan]")
    df = df.reset_index(drop=True).copy()
    
    # 检查 MFA 是否可用
    if not check_mfa_available():
        rprint("[yellow]⚠️ MFA 未安装或不可用，跳过对齐优化[/yellow]")
        rprint("[yellow]   请运行 python install_mfa.py 安装 MFA[/yellow]")
        return df
    
    # 读取配置
    acoustic_model = _load_mfa_config("mfa.acoustic_model", "english_mfa")
    dictionary = _load_mfa_config("mfa.dictionary", "english_mfa")
    
    # 创建临时工作目录
    work_dir = tempfile.mkdtemp(prefix='mfa_')
    output_dir = os.path.join(work_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        source_audio = AudioSegment.from_file(audio_file)
        original_df = df.copy()

        # 1. 准备输入
        audio_dest, txt_path = prepare_mfa_input(df, audio_file, work_dir)
        input_dir = os.path.dirname(audio_dest)
        
        # 2. 运行 MFA 对齐
        success = run_mfa_alignment(input_dir, output_dir, acoustic_model, dictionary)
        
        if not success:
            rprint("[yellow]⚠️ MFA 对齐失败，使用原始时间戳[/yellow]")
            return df
        
        # 3. 解析 TextGrid
        textgrid_files = [f for f in os.listdir(output_dir) if f.endswith('.TextGrid')]
        if not textgrid_files:
            rprint("[yellow]⚠️ 未找到 MFA 输出文件，使用原始时间戳[/yellow]")
            return df
        
        textgrid_path = os.path.join(output_dir, textgrid_files[0])
        mfa_words = parse_textgrid(textgrid_path)
        
        if not mfa_words:
            rprint("[yellow]⚠️ TextGrid 解析失败，使用原始时间戳[/yellow]")
            return df
        
        rprint(f"[cyan]📊 MFA 输出: {len(mfa_words)} 个词[/cyan]")
        
        # 4. 先应用全局 MFA 结果，再对可疑大偏移和重叠块做局部重对齐
        df, global_matches = update_timestamps(df, mfa_words)
        df, late_repaired_count = _repair_late_start_shifts(
            df=df,
            original_df=original_df,
            global_matches=global_matches,
            source_audio=source_audio,
            acoustic_model=acoustic_model,
            dictionary=dictionary,
            work_dir=work_dir,
        )
        df, overlap_repaired_count = _repair_overlap_blocks(
            df=df,
            original_df=original_df,
            source_audio=source_audio,
            acoustic_model=acoustic_model,
            dictionary=dictionary,
            work_dir=work_dir,
        )

        if late_repaired_count or overlap_repaired_count:
            rprint(
                f"[green]✅ 局部 MFA 修正汇总: late-shift repaired={late_repaired_count}, "
                f"overlap repaired={overlap_repaired_count}[/green]"
            )
        
        rprint("[green]✅ MFA 对齐完成[/green]")
        return df
        
    except Exception as e:
        rprint(f"[red]❌ MFA 对齐错误: {e}[/red]")
        rprint("[yellow]   使用原始时间戳[/yellow]")
        return df
        
    finally:
        # 清理临时文件
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass
