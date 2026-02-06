import os
import pandas as pd
import warnings
from core.spacy_utils.load_nlp_model import init_nlp, ROUGH_SPLIT_FILE
from core.utils.config_utils import load_key, get_joiner
from rich import print as rprint

warnings.filterwarnings("ignore", category=FutureWarning)


def rough_split(nlp):
    """
    按标点分句：
    1. 先根据时间间隔把文本分成多个段落
    2. 对每个段落用 spacy 按标点分句
    这样既保留了 spacy 的智能分句，又能在时间断点处强制分开。
    """
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    rprint(f"[blue]🔍 Using {language} language joiner: '{joiner}'[/blue]")
    
    # 读取时间间隔阈值
    time_gap_threshold = load_key("subtitle.time_split_threshold")
    rprint(f"[blue]⏱️ Time gap threshold: {time_gap_threshold}s[/blue]")
    
    chunks = pd.read_excel("output/log/cleaned_chunks.xlsx")
    chunks['text'] = chunks['text'].apply(lambda x: str(x).strip('"').strip())
    
    # 第一步：根据时间间隔分成多个段落
    paragraphs = []  # 每个段落是一个词列表
    current_paragraph = []
    prev_end_time = None
    
    for idx, row in chunks.iterrows():
        word = row['text']
        start_time = row['start']
        end_time = row['end']
        
        if not word or str(word).isspace():
            continue
        
        # 检查是否需要开始新段落
        if prev_end_time is not None:
            time_gap = start_time - prev_end_time
            if time_gap > time_gap_threshold:
                # 保存当前段落，开始新段落
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                current_paragraph = []
                rprint(f"[dim]📍 Time gap {time_gap:.2f}s at {start_time:.2f}s[/dim]")
        
        current_paragraph.append(word)
        prev_end_time = end_time
    
    # 保存最后一个段落
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    rprint(f"[blue]📍 Split into {len(paragraphs)} paragraphs based on time gaps[/blue]")
    
    # 第二步：对每个段落用 spacy 分句
    all_sentences = []
    
    for paragraph_words in paragraphs:
        # 用 joiner 拼接段落内的词
        paragraph_text = joiner.join(paragraph_words)
        
        # 用 spacy 分句
        doc = nlp(paragraph_text)
        
        if not doc.has_annotation("SENT_START"):
            # 如果 spacy 无法分句，保留整个段落
            all_sentences.append(paragraph_text)
            continue
        
        # 处理 spacy 分出的句子（合并 - 和 ... 开头/结尾的情况）
        current_sentence = []
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            
            if current_sentence and (
                text.startswith('-') or 
                text.startswith('...') or
                current_sentence[-1].endswith('-') or
                current_sentence[-1].endswith('...')
            ):
                current_sentence.append(text)
            else:
                if current_sentence:
                    all_sentences.append(' '.join(current_sentence))
                    current_sentence = []
                current_sentence.append(text)
        
        if current_sentence:
            all_sentences.append(' '.join(current_sentence))

    # 写入文件
    with open(ROUGH_SPLIT_FILE, "w", encoding="utf-8") as output_file:
        for i, sentence in enumerate(all_sentences):
            if i > 0 and sentence.strip() in [',', '.', '，', '。', '？', '！']:
                # 如果当前行只有标点，合并到上一行
                output_file.seek(output_file.tell() - 1, os.SEEK_SET)
                output_file.write(sentence)
            else:
                output_file.write(sentence + "\n")
    
    rprint(f"[green]✅ Split into {len(all_sentences)} sentences[/green]")
    rprint(f"[green]💾 Saved to → `{ROUGH_SPLIT_FILE}`[/green]")

if __name__ == "__main__":
    nlp = init_nlp()
    rough_split(nlp)
