import pandas as pd
import json
import concurrent.futures
from core.translate_lines import translate_lines
from core.prompts import get_prompt_single_pass_full_polish
from core.summarizer import search_things_to_note_in_prompt
from core.utils.text_trim import check_len_then_trim
from core.subtitle_generator import align_timestamp
from core.utils import *
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from difflib import SequenceMatcher
from core.utils.paths import *
console = Console()

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

def polish_single_pass_full_text(src_lines, draft_lines, summary_prompt=None):
    if len(src_lines) != len(draft_lines):
        raise ValueError("Full polish input mismatch: source and translation line counts are different.")

    line_count = len(draft_lines)
    prompt = get_prompt_single_pass_full_polish(src_lines, draft_lines, summary_prompt)
    required_keys = [str(i) for i in range(1, line_count + 1)]

    def valid_full_polish(response_data):
        if not isinstance(response_data, dict):
            return {"status": "error", "message": "Response is not a JSON object."}

        missing_keys = [k for k in required_keys if k not in response_data]
        if missing_keys:
            return {
                "status": "error",
                "message": f"Missing required key(s): {', '.join(missing_keys[:10])}",
            }

        for i in range(1, line_count + 1):
            key = str(i)
            item = response_data.get(key)
            if not isinstance(item, dict):
                return {"status": "error", "message": f"Invalid item format at key {key}."}
            if "free" not in item:
                return {"status": "error", "message": f"Missing `free` in item {key}."}

            polished = str(item["free"]).replace("\n", " ").strip()
            src = str(src_lines[i - 1]).strip()
            if not polished and src:
                return {"status": "error", "message": f"Empty polished line at key {key}."}

        return {"status": "success", "message": "Full polish completed"}

    api_settings = _load_single_pass_full_polish_api_settings()
    result = ask_gpt(
        prompt,
        resp_type="json",
        valid_def=valid_full_polish,
        log_title="single_pass_full_polish",
        api_settings=api_settings,
    )
    polished_lines = [
        str(result[str(i)]["free"]).replace("\n", " ").strip()
        for i in range(1, line_count + 1)
    ]

    if len(polished_lines) != line_count:
        raise ValueError("Full polish output mismatch: line count differs from input.")

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
    console.print("[bold green]✅ Translation completed and results saved.[/bold green]")

if __name__ == '__main__':
    translate_all()
