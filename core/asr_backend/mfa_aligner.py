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
from typing import List, Tuple
from rich import print as rprint
from core.utils import load_key

MAX_WORD_CENTER_SHIFT_SEC = 0.5

def _load_mfa_config(key: str, default):
    try:
        value = load_key(key)
    except Exception:
        return default

    if value in (None, ""):
        return default
    return value

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


def _is_center_shift_allowed(
    original_start: float,
    original_end: float,
    new_start: float,
    new_end: float,
    max_center_shift_sec: float = MAX_WORD_CENTER_SHIFT_SEC,
) -> bool:
    original_center = (float(original_start) + float(original_end)) / 2.0
    new_center = (float(new_start) + float(new_end)) / 2.0
    return abs(new_center - original_center) <= float(max_center_shift_sec)

def update_timestamps(df: pd.DataFrame, mfa_words: List[Tuple[str, float, float]]) -> pd.DataFrame:
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
    
    # 清理 stable-ts 的文本（去除引号）
    df['clean_text'] = df['text'].apply(lambda x: str(x).strip('"').strip("'").strip().lower())
    
    # MFA 词列表（小写用于匹配）
    mfa_lower = [(w.lower(), s, e) for w, s, e in mfa_words]
    
    updated_count = 0
    rejected_count = 0
    mfa_idx = 0
    
    for i, row in df.iterrows():
        if mfa_idx >= len(mfa_lower):
            break
        
        stable_word = row['clean_text']
        original_start = float(row['start'])
        original_end = float(row['end'])

        matched = False
        for skip_count in range(4):
            if mfa_idx + skip_count >= len(mfa_lower):
                break

            check_word, check_start, check_end = mfa_lower[mfa_idx + skip_count]
            if stable_word == check_word or stable_word in check_word or check_word in stable_word:
                if _is_center_shift_allowed(
                    original_start=original_start,
                    original_end=original_end,
                    new_start=check_start,
                    new_end=check_end,
                ):
                    df.at[i, 'start'] = check_start
                    df.at[i, 'end'] = check_end
                    updated_count += 1
                    mfa_idx = mfa_idx + skip_count + 1
                    matched = True
                    break

                rejected_count += 1
                break

        if not matched:
            continue
    
    # 清理临时列
    df = df.drop(columns=['clean_text'])
    
    rprint(
        f"[green]✅ MFA 时间戳更新: {updated_count}/{len(df)} 个词[/green]"
        + (f" [yellow](rejected by center clamp: {rejected_count})[/yellow]" if rejected_count else "")
    )
    
    return df

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
        
        # 4. 更新时间戳
        df = update_timestamps(df, mfa_words)
        
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
