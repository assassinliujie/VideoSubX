import json
from core.utils import *

## ================================================================
# @ step4_splitbymeaning.py
def get_split_prompt(sentence, num_parts = 2, word_limit = 20):
    language = load_key("whisper.detected_language")
    split_prompt = f"""
## Role
You are a professional Netflix subtitle splitter in **{language}**.

## Task
Split the given subtitle text into **{num_parts}** parts, each less than **{word_limit}** words.

1. Maintain sentence meaning coherence according to Netflix subtitle standards, and ensure each part is a meaningful phrase/clause unit (not a dangling fragment).
2. MOST IMPORTANT: Do not end a non-final part with dangling function words, especially prepositions/conjunctions such as: of, into, from, to, for, with, at, on, in, by, about, as, and, or.
3. Avoid unnecessary splitting: if the sentence is already natural and does not need splitting (while still meeting length limits), prefer keeping it unsplit.
4. Keep parts roughly equal in length (minimum 3 words each) when possible, but this is lower priority than Rule 1-3.
5. Split at natural points like punctuation marks or conjunctions.
6. If provided text is repeated words, simply split at the middle of the repeated words.

### Text Fidelity Constraint (STRICT)
1. Keep the source wording exactly as-is.
2. Do NOT rewrite grammar, paraphrase, or "improve" intentional jokes/mistakes.
3. Do NOT normalize colloquial spoken forms, including but not limited to: gonna, wanna, gotta, kinda, sorta, ain't, y'all.
4. Outside `[br]`, text must be character-preserving.

Priority note: If constraints conflict, prioritize semantic integrity and no-dangling-endings first, then avoid unnecessary splitting, and finally optimize length balance.

## Steps
1. Analyze sentence structure, complexity, and splitting challenges.
2. Generate two alternative splitting approaches with [br] tags at split positions.
3. Compare both approaches highlighting their strengths and weaknesses.
4. Choose the best splitting approach.

## Given Text
<split_this_sentence>
{sentence}
</split_this_sentence>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief description of sentence structure, complexity, and key splitting challenges",
    "split1": "First splitting approach with [br] tags at split positions",
    "split2": "Alternative splitting approach with [br] tags at split positions",
    "assess": "Comparison of both approaches highlighting their strengths and weaknesses",
    "choice": "1 or 2"
}}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()
    return split_prompt

def get_english_correction_prompt(tokens_json: str):
    return f"""
## Role
You are an English ASR token correction expert.

## Task
Given word-level English ASR tokens, identify only the tokens that should be corrected.

English correction scope: fix spelling or ASR errors, including misspelled proper nouns (person/brand/company/product names). You may only replace existing tokens; do not add, delete, or reorder tokens. No other modifications to the English text are allowed.

## Decision Policy (Conservative, default = keep original)
- Default action is KEEP the token unchanged.
- Only correct when the source token is clearly wrong and the target is uniquely supported by local context.
- If there is any ambiguity, do not correct.

## Rules
1. You must use `start_key` as the primary key for each correction.
2. Only return high-confidence corrections.
3. Do NOT add or delete tokens.
4. Do NOT reorder tokens.
5. Do NOT rewrite grammar or style.
6. Do NOT normalize colloquial forms (e.g., gonna/wanna/gotta/kinda/sorta/ain't/y'all) under any circumstance.
7. If uncertain, do not correct.
8. If a token is a plausible acronym/proper noun/style token (e.g., ALL CAPS, TitleCase, mixed model tokens), do NOT "normalize" it to a more common word unless context is explicit and unambiguous.
9. Never replace one plausible acronym with another plausible acronym based on guesswork (e.g., LAN <-> WAN). If both are plausible, keep original.
10. Brand/event/product names may intentionally use uncommon spellings; do not auto-correct those based only on dictionary frequency or phonetic similarity.
11. Do NOT output no-op suggestions. If `source` equals `target`, that item must be omitted.

## INPUT
<tokens_json>
{tokens_json}
</tokens_json>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of correction confidence and error types",
    "corrections": [
        {{
            "start_key": "exact start_key from input",
            "source": "original token",
            "target": "corrected token",
            "type": "spelling|asr|person|brand|company|product",
            "confidence": "high",
            "reason": "brief reason"
        }}
    ]
}}
```

If no corrections are needed, return `"corrections": []`.
Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()


def get_rough_split_entity_repair_prompt(boundary_pairs_json: str):
    return f"""
## Role
You are a subtitle boundary quality checker.

## Task
Given adjacent subtitle line pairs from rough splitting, detect ONLY the cases where a multi-word proper noun or named entity is broken across the line boundary.

A valid correction means:
- the entity is composed of a suffix of the left line + a prefix of the right line
- both parts are contiguous at the boundary
- confidence is high

## Rules
1. Scope is strict: proper nouns / named entities only.
   Types include: person, company, organization, product/model, place, title, event.
2. Do NOT propose grammatical/style rewrites.
3. Do NOT propose fixes for generic phrases or common collocations.
4. Return high-confidence items only. If unsure, skip.
5. `left_words` and `right_words` are positive integers and must refer to boundary words only.
6. Keep fragments short (usually 1-4 words per side).
7. Do not return duplicate corrections for the same `pair_id`.

## INPUT
<boundary_pairs_json>
{boundary_pairs_json}
</boundary_pairs_json>

## Output in only JSON format and no other text
```json
{{
  "analysis": "Brief summary of boundary quality and confidence",
  "corrections": [
    {{
      "pair_id": 12,
      "left_words": 1,
      "right_words": 2,
      "entity": "5070 Ti Super",
      "type": "product",
      "confidence": "high",
      "reason": "GPU model suffix split across boundary"
    }}
  ]
}}
```

If no correction is needed, return `"corrections": []`.
Note: Start your answer with ```json and end with ```, do not add any other text.
""".strip()

"""{{
    "analysis": "Brief analysis of the text structure",
    "split": "Complete sentence with [br] tags at split positions"
}}"""

## ================================================================
# @ step4_1_summarize.py
def get_summary_prompt(source_content, custom_terms_json=None):
    src_lang = load_key("whisper.detected_language")
    tgt_lang = load_key("target_language")
    
    # add custom terms note
    terms_note = ""
    if custom_terms_json:
        terms_list = []
        for term in custom_terms_json['terms']:
            terms_list.append(f"- {term['src']}: {term['tgt']} ({term['note']})")
        terms_note = "\n### Existing Terms\nPlease exclude these terms in your extraction:\n" + "\n".join(terms_list)
    
    summary_prompt = f"""
## Role
You are a video translation expert and terminology consultant, specializing in {src_lang} comprehension and {tgt_lang} expression optimization.

## Task
For the provided {src_lang} video text:
1. Summarize main topic in two sentences
2. Extract professional terms/names with {tgt_lang} translations (excluding existing terms)
3. Provide brief explanation for each term

{terms_note}

Steps:
1. Topic Summary:
    - Quick scan for general understanding
    - Write two sentences: first for main topic, second for key point
2. Term Extraction:
    - Mark professional terms and names (excluding those listed in Existing Terms)
    - Provide {tgt_lang} translation or keep original
    - Add brief explanation
    - Extract less than 15 terms

## INPUT
<text>
{source_content}
</text>

## Output in only JSON format and no other text
{{
  "theme": "Two-sentence video summary",
  "terms": [
    {{
      "src": "{src_lang} term",
      "tgt": "{tgt_lang} translation or original", 
      "note": "Brief explanation"
    }},
    ...
  ]
}} 

## Example
{{
  "theme": "本视频介绍人工智能在医疗领域的应用现状。重点展示了AI在医学影像诊断和药物研发中的突破性进展。",
  "terms": [
    {{
      "src": "Machine Learning",
      "tgt": "机器学习",
      "note": "AI的核心技术，通过数据训练实现智能决策"
    }},
    {{
      "src": "CNN",
      "tgt": "CNN",
      "note": "卷积神经网络，用于医学图像识别的深度学习模型"
    }}
  ]
}}

Note: Start you answer with ```json and end with ```, do not add any other text.
""".strip()
    return summary_prompt

## ================================================================
# @ step5_translate.py & translate_lines.py
def generate_shared_prompt(previous_content_prompt, after_content_prompt, summary_prompt, things_to_note_prompt):
    return f'''### Context Information
<previous_content>
{previous_content_prompt}
</previous_content>

<subsequent_content>
{after_content_prompt}
</subsequent_content>

### Content Summary
{summary_prompt}

### Points to Note
{things_to_note_prompt}'''

def get_prompt_faithfulness(lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    # Split lines by \n
    line_splits = lines.split('\n')
    
    json_dict = {}
    for i, line in enumerate(line_splits, 1):
        json_dict[f"{i}"] = {"origin": line, "direct": f"direct {TARGET_LANGUAGE} translation {i}."}
    json_format = json.dumps(json_dict, indent=2, ensure_ascii=False)

    src_language = load_key("whisper.detected_language")
    prompt_faithfulness = f'''
## Role
You are a professional Netflix subtitle translator, fluent in both {src_language} and {TARGET_LANGUAGE}, as well as their respective cultures. 
Your expertise lies in accurately understanding the semantics and structure of the original {src_language} text and faithfully translating it into {TARGET_LANGUAGE} while preserving the original meaning.

## Task
We have a segment of original {src_language} subtitles that need to be directly translated into {TARGET_LANGUAGE}. These subtitles come from a specific context and may contain specific themes and terminology.

1. Translate the original {src_language} subtitles into {TARGET_LANGUAGE} line by line
2. Ensure the translation is faithful to the original, accurately conveying the original meaning
3. Consider the context and professional terminology

{shared_prompt}

<translation_principles>
1. Faithful to the original: Accurately convey the content and meaning of the original text, without arbitrarily changing, adding, or omitting content.
2. Accurate terminology: Use professional terms correctly and maintain consistency in terminology.
3. Understand the context: Fully comprehend and reflect the background and contextual relationships of the text.
</translation_principles>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_faithfulness.strip()


def get_prompt_expressiveness(faithfulness_result, lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    json_format = {
        key: {
            "origin": value["origin"],
            "direct": value["direct"],
            "reflect": "your reflection on direct translation",
            "free": "your free translation"
        }
        for key, value in faithfulness_result.items()
    }
    json_format = json.dumps(json_format, indent=2, ensure_ascii=False)

    src_language = load_key("whisper.detected_language")
    prompt_expressiveness = f'''
## Role
You are a professional Netflix subtitle translator and language consultant.
Your expertise lies not only in accurately understanding the original {src_language} but also in optimizing the {TARGET_LANGUAGE} translation to better suit the target language's expression habits and cultural background.

## Task
We already have a direct translation version of the original {src_language} subtitles.
Your task is to reflect on and improve these direct translations to create more natural and fluent {TARGET_LANGUAGE} subtitles.

### Core Principle: Structural Integrity
**CRITICAL CONSTRAINT**: The number of translated lines must exactly match the number of original lines. Never merge the meaning of two lines into a single translated line if it results in another line becoming empty. Every original line must have a corresponding, non-empty translation.

### Additional Optimization Guidelines:
1.  **Semantic Distribution and Redundancy Elimination**:
    - **Problem**: A common AI error is when translating adjacent source lines (e.g., Line A and Line B), the translation for Line A improperly contains the combined meaning of A+B. Then, the translation for Line B unnecessarily repeats the meaning of B, creating redundancy.
    - **Your Task**: Identify and correct this. Instead of merging, you must **redistribute the semantic components** logically and naturally across the corresponding translated lines. The goal is a smooth, non-repetitive flow where both lines contribute meaningfully.
    - **Example of Redundancy**:
        - Origin A: `We need to analyze the economic data,`
        - Origin B: `which is an absolutely critical step.`
        - Bad (Redundant): A: `我们需要分析至关重要的经济数据` B: `这是非常关键的一步` (The concept of "critical" is repeated in both lines).
        - Good (Distributed): A: `我们需要分析经济数据，` B: `这是至关重要的一步。` (The meaning is correctly distributed).

2.  **Completeness of Detail**: While eliminating redundancy, you must ensure that no specific details, examples, or nuances from the original text are lost. Every key piece of information (such as quantities, examples, conditions, or descriptive adjectives) must be accurately reflected in the final translation, even if it requires slightly longer or more complex phrasing.

3.  **Natural Sentence Splitting**: When the original text splits a single idea across adjacent lines (e.g., due to a speaker's pause), the translation must also be split naturally across the corresponding lines.
    - **Prohibition**: Do not merge the full translation into the first line while leaving the second line's translation empty.
    - **Recommended Technique**: Create a smooth, natural break that respects {TARGET_LANGUAGE} grammar. The first line can end as a natural fragment if the thought is clearly completed in the second line. Punctuation should be used to create a natural flow, not to signal an incomplete sentence with an ellipsis unless stylistically appropriate for the context.

4.  **Word Order Flexibility**: Within a batch of lines, you are encouraged to reorder or swap translated sentence components across adjacent lines to achieve a more natural {TARGET_LANGUAGE} expression flow, as long as the line count is maintained and the overall meaning is preserved.

5.  **Fact Checking**: Ensure proper nouns, brand names, technical terms, and cultural references are accurately translated. Double-check for any potential misinterpretations.

6.  **Length Optimization**: Aim for concise translations that fit comfortable reading time (around 15 Chinese characters per line when possible), while preserving meaning.

7.  **Cultural Localization**: Use appropriate Chinese expressions, idioms, or colloquialisms that feel natural to native speakers without forced localization.

8.  **Oral Connectors**: Handle English connectors like "but", "so", "well" appropriately - avoid rigid translations when they serve as verbal fillers rather than logical connectors.
    
9.  **Numerals**: Use Chinese numerals (〇一二三四) for small numbers and emphasis, Arabic numerals for technical content, dates, and larger numbers.

### Translation Process:
1. Analyze direct translations line by line
2. Identify optimization opportunities within the batch context
3. Apply natural Chinese expression patterns
4. Ensure factual accuracy and cultural appropriateness

{shared_prompt}

<Translation Analysis Steps>
Please use a contextual approach to optimize the text:

1. Direct Translation Reflection:
    - Evaluate language fluency and cultural appropriateness
    - Check for factual accuracy (names, terms, references)
    - Identify opportunities for word order adjustment within the batch

2. {TARGET_LANGUAGE} Free Translation:
    - Aim for contextual smoothness and naturalness
    - Apply appropriate cultural adaptations
    - Ensure conciseness while preserving meaning
    - Maintain coherence between adjacent lines
</Translation Analysis Steps>
    
## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_expressiveness.strip()

def get_prompt_single_pass(lines, shared_prompt):
    TARGET_LANGUAGE = load_key("target_language")
    line_splits = lines.split('\n')
    json_format = {}
    for i, line in enumerate(line_splits, 1):
        json_format[f"{i}"] = {
            "origin": line,
            "direct": f"faithful {TARGET_LANGUAGE} translation {i}",
            "reflect": "brief reflection on wording and structure",
            "free": f"natural and concise {TARGET_LANGUAGE} subtitle {i}",
        }
    json_format = json.dumps(json_format, indent=2, ensure_ascii=False)

    src_language = load_key("whisper.detected_language")
    prompt_single_pass = f'''
## Role
You are a professional Netflix subtitle translator and language consultant.
You are fluent in both {src_language} and {TARGET_LANGUAGE}, as well as their respective cultures.
Your expertise lies in accurately understanding the semantics and structure of the original text, then optimizing it for natural subtitle reading.

## Task
Translate the original {src_language} subtitles into high-quality {TARGET_LANGUAGE} subtitles in a single pass.
For each line, provide:
1. `direct`: a faithful translation that preserves original meaning and details
2. `reflect`: a brief reflection on wording and structure improvements
3. `free`: a final natural subtitle line optimized for readability

### Phase A: Faithfulness (must satisfy first)
1. Faithful to the original: accurately convey the original meaning without arbitrary additions or omissions.
2. Accurate terminology: use professional terms correctly and consistently.
3. Context awareness: fully reflect the background and contextual relationships.

### Phase B: Expressiveness (optimize after faithfulness)
1.  **Semantic Distribution and Redundancy Elimination**:
    - **Problem**: A common AI error is when translating adjacent source lines (e.g., Line A and Line B), the translation for Line A improperly contains the combined meaning of A+B. Then, the translation for Line B unnecessarily repeats the meaning of B, creating redundancy.
    - **Your Task**: Identify and correct this. Instead of merging, you must **redistribute the semantic components** logically and naturally across the corresponding translated lines. The goal is a smooth, non-repetitive flow where both lines contribute meaningfully.
2.  **Completeness of Detail**: While eliminating redundancy, ensure no specific details, examples, or nuances are lost.
3.  **Natural Sentence Splitting**: Keep natural split flow when one idea spans adjacent source lines.
4.  **Word Order Flexibility**: You may reorder components across adjacent lines for natural flow, while preserving line count and meaning.
5.  **Fact Checking**: Keep proper nouns, brand names, technical terms, and cultural references accurate.
6.  **Length Optimization**: Keep subtitles concise and readable.
7.  **Cultural Localization**: Use natural target-language expressions without forced localization.
8.  **Oral Connectors**: Handle connectors like "but", "so", "well" naturally.
9.  **Numerals**: Use Chinese numerals (〇一二三四) for small numbers and emphasis; use Arabic numerals for technical content, dates, and larger numbers.

### Core Structural Constraints (hard constraints)
1. Line count must exactly match the number of source lines.
2. Never leave empty translations.
3. Never merge two source lines into one target line.
4. Every source line must have a corresponding non-empty target line.

{shared_prompt}

<Translation Analysis Steps>
1. Produce `direct` first (faithful, precise, complete).
2. Briefly reflect in `reflect` (fluency, factual checks, structural issues).
3. Produce `free` as final subtitle line (natural, concise, context-aware).
</Translation Analysis Steps>

## INPUT
<subtitles>
{lines}
</subtitles>

## Output in only JSON format and no other text
```json
{json_format}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''
    return prompt_single_pass.strip()

def get_prompt_single_pass_full_polish(source_lines, draft_lines, summary_prompt=None):
    TARGET_LANGUAGE = load_key("target_language")
    src_language = load_key("whisper.detected_language")
    line_count = len(draft_lines)

    source_block = "\n".join(
        f"[{i}] {str(line)}" for i, line in enumerate(source_lines, 1)
    )
    draft_block = "\n".join(
        f"[{i}] {str(line)}" for i, line in enumerate(draft_lines, 1)
    )

    json_format = """{
  "1": {"free": "第1行润色结果"},
  "2": {"free": "第2行润色结果"},
  "...": {"free": "..."},
  "N": {"free": "第N行润色结果"}
}"""
    summary_prompt = summary_prompt if summary_prompt else "N/A"

    prompt = f'''
## Role
你是资深中文字幕本地化润色专家，熟悉中英双语语义和中文日常表达习惯。

## Task
给定“全文{src_language}原文分行”和“全文{TARGET_LANGUAGE}草译分行”，请对{TARGET_LANGUAGE}草译做一次全文统一润色。
目标是提升自然度、连贯性、可读性与术语一致性，同时严格保持逐行对齐。

### Hard Constraints
1. 必须全文一次性处理，但输出必须逐行对应输入行号。
2. 输出总行数必须与输入完全一致（共 {line_count} 行）。
3. 严禁合并、拆分、重排、增删行。
4. 每一行输出必须是单行文本，禁止在行内再换行。
5. 对应原文非空的行，译文不得为空。
6. 不允许改变原文事实、立场、褒贬色彩，不允许过度引申或过度翻译。
7. 专有名词（人名、地名、机构名、产品名等）译法必须全文统一。
8. 同一术语和同类表达必须全文用词统一。
9. 不要对{src_language}原文做任何改写或纠错；本任务只润色{TARGET_LANGUAGE}译文。

### Style and Quality Rules
1. 允许按中文习惯调整语序（含倒装、局部换位、跨短句重组），但不得破坏行对齐约束。
2. 优先保证“信、达、雅”：忠实原意、表达通顺、风格自然。
3. 可适度补充中文语气连接（如“则/那/故/竟”等）以增强连贯性，但不得凭空添加信息。
4. 对英文口语连接词（如 but/so）若仅为口头衔接，不要机械译成“但/所以/然而”。
5. 可适度使用地道中文表达（含成语）增强本地化，但避免生硬堆砌。
6. 字幕应尽量简洁，单行长度以“易读”为优先，理想约 15 字左右（软约束，不得因压缩而丢信息）。
7. 标点规则：除问号、感叹号、引号外，其余标点尽量弱化处理（必要时用空格替代），保持画面阅读简洁。

### Content Summary
{summary_prompt}

## INPUT
<source_subtitles>
{source_block}
</source_subtitles>

<draft_translation>
{draft_block}
</draft_translation>

## Output in only JSON format and no other text
仅输出 JSON，不要输出任何解释或额外文本。
JSON 顶层键必须为字符串数字 "1" 到 "{line_count}"，每项格式如下：
```json
{json_format}
```

Note: Start your answer with ```json and end with ```, do not add any other text.
'''.strip()

    return prompt


## ================================================================
# @ step6_splitforsub.py
def get_align_prompt(src_sub, tr_sub, src_part):
    targ_lang = load_key("target_language")
    src_lang = load_key("whisper.detected_language")
    src_splits = src_part.split('\n')
    num_parts = len(src_splits)
    src_part = src_part.replace('\n', ' [br] ')
    align_parts_json = ','.join(
        f'''
        {{
            "src_part_{i+1}": "{src_splits[i]}",
            "target_part_{i+1}": "Corresponding aligned {targ_lang} subtitle part"
        }}''' for i in range(num_parts)
    )

    align_prompt = f'''
## Role
You are a Netflix subtitle alignment expert fluent in both {src_lang} and {targ_lang}.

## Task
We have {src_lang} and {targ_lang} original subtitles for a Netflix program, as well as a pre-processed split version of {src_lang} subtitles.
Your task is to create the best splitting scheme for the {targ_lang} subtitles based on this information.

1. Analyze the word order and structural correspondence between {src_lang} and {targ_lang} subtitles
2. Split the {targ_lang} subtitles according to the pre-processed {src_lang} split version
3. Never leave empty lines. If it's difficult to split based on meaning, you may appropriately rewrite the sentences that need to be aligned
4. Do not add comments or explanations in the translation, as the subtitles are for the audience to read

## INPUT
<subtitles>
{src_lang} Original: "{src_sub}"
{targ_lang} Original: "{tr_sub}"
Pre-processed {src_lang} Subtitles ([br] indicates split points): {src_part}
</subtitles>

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of word order, structure, and semantic correspondence between two subtitles",
    "align": [
        {align_parts_json}
    ]
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
    return align_prompt

## ================================================================
# @ step8_gen_audio_task.py @ step10_gen_audio.py
def get_subtitle_trim_prompt(text, duration):

    rule = '''Consider a. Reducing filler words without modifying meaningful content. b. Omitting unnecessary modifiers or pronouns, for example:
    - "Please explain your thought process" can be shortened to "Please explain thought process"
    - "We need to carefully analyze this complex problem" can be shortened to "We need to analyze this problem"
    - "Let's discuss the various different perspectives on this topic" can be shortened to "Let's discuss different perspectives on this topic"
    - "Can you describe in detail your experience from yesterday" can be shortened to "Can you describe yesterday's experience" '''

    trim_prompt = f'''
## Role
You are a professional subtitle editor, editing and optimizing lengthy subtitles that exceed voiceover time before handing them to voice actors. 
Your expertise lies in cleverly shortening subtitles slightly while ensuring the original meaning and structure remain unchanged.

## INPUT
<subtitles>
Subtitle: "{text}"
Duration: {duration} seconds
</subtitles>

## Processing Rules
{rule}

## Processing Steps
Please follow these steps and provide the results in the JSON output:
1. Analysis: Briefly analyze the subtitle's structure, key information, and filler words that can be omitted.
2. Trimming: Based on the rules and analysis, optimize the subtitle by making it more concise according to the processing rules.

## Output in only JSON format and no other text
```json
{{
    "analysis": "Brief analysis of the subtitle, including structure, key information, and potential processing locations",
    "result": "Optimized and shortened subtitle in the original subtitle language"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
    return trim_prompt

## ================================================================
# @ tts_main
def get_correct_text_prompt(text):
    return f'''
## Role
You are a text cleaning expert for TTS (Text-to-Speech) systems.

## Task
Clean the given text by:
1. Keep only basic punctuation (.,?!)
2. Preserve the original meaning

## INPUT
{text}

## Output in only JSON format and no other text
```json
{{
    "text": "cleaned text here"
}}
```

Note: Start you answer with ```json and end with ```, do not add any other text.
'''.strip()
