import json
from core.utils import *

## ================================================================
# @ step4_splitbymeaning.py
def get_split_prompt(sentence, num_parts = 2, word_limit = 20):
    language = load_key("whisper.detected_language")
    split_prompt = f"""
## 角色
你是一名处理 **{language}** 字幕的专业 Netflix 字幕分句专家。

## 任务
在给定字幕原文中插入 `[br]` 标记，按照自然语义边界将文本拆成合适数量的部分。

长度计算给出的参考部分数为 **{num_parts}**，每个部分必须少于 **{word_limit}** 个词。实际部分数根据语义边界和长度上限确定，输出必须至少包含一个 `[br]`。

## 分句规则
1. 遵循 Netflix 字幕标准，保持整句语义连贯。每个部分都应构成有意义且相对完整的短语、短句或分句，避免形成悬空残片。
2. 最重要：非最后一个部分的末尾不得留下悬挂的功能词，尤其是介词或连词，例如：of、into、from、to、for、with、at、on、in、by、about、as、and、or。
3. 普通情况下倾向拆成两个部分，并避免不必要的过度拆分。
4. 一句话包含多个相对独立的短句或分句时，尽量在自然语义边界拆开，可以拆成三到四部分。
5. 三到四部分属于建议范围。极长文本可以根据长度要求拆成更多部分，每个部分仍须少于 **{word_limit}** 个词。
6. 条件允许时，各部分长度尽量均衡，每部分至少包含 3 个词。语义完整和非悬挂结尾具有更高优先级。
7. 优先选择标点符号、连词附近或其他自然停顿位置作为断点。
8. 输入文本由重复词组成时，在重复词序列的中间位置断开。
9. 在满足每部分长度上限的前提下，约束发生冲突时依次保证：语义完整和非悬挂结尾、避免过度拆分、长度均衡。

## 原文保真约束（严格）
1. 完整保留原文措辞。
2. 只允许插入 `[br]`，不得增加、删除、替换或调整其他字符的顺序。
3. 保留原文中的标点、大小写、空格和所有其他字符。
4. 不修改语法，不进行改写、释义或润色，也不修正原文中有意保留的笑话或错误。
5. 保留所有口语形式，包括但不限于：gonna、wanna、gotta、kinda、sorta、ain't、y'all。
6. 除新增的 `[br]` 标记外，输出文本必须与输入文本逐字符一致。

## 待分句文本
<split_this_sentence>
{sentence}
</split_this_sentence>

## 输出要求
仅输出一个可解析的 JSON 对象，不使用 Markdown 代码块，不添加解释或其他文字。JSON 对象只包含必填字符串字段 `split`：

{{
  "split": "在原始文本的自然断点处插入 [br] 后得到的完整文本"
}}

`split` 中必须包含一个或多个 `[br]`。
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

def get_prompt_single_pass_full_polish(
    source_lines,
    draft_lines,
    summary_prompt=None,
    start_line=1,
    polished_prefix_lines=None,
):
    TARGET_LANGUAGE = load_key("target_language")
    src_language = load_key("whisper.detected_language")
    line_count = len(draft_lines)
    start_line = max(1, int(start_line))
    polished_prefix_lines = polished_prefix_lines or []

    remaining_source_block = "\n".join(
        f"[{i}] {str(line)}"
        for i, line in enumerate(source_lines[start_line - 1 :], start_line)
    )
    remaining_draft_block = "\n".join(
        f"[{i}] {str(line)}"
        for i, line in enumerate(draft_lines[start_line - 1 :], start_line)
    )
    polished_prefix_block = "\n".join(
        f"[{i}] {str(line)}"
        for i, line in enumerate(polished_prefix_lines[: start_line - 1], 1)
    )
    summary_prompt = summary_prompt if summary_prompt else "N/A"

    if start_line > 1:
        resume_notice = f"""
### Resume Notice
1. 前 {start_line - 1} 行已润色完成，仅作为术语、语气、文风和上下文参考。
2. 不要重写、重复输出或改写第 1 行到第 {start_line - 1} 行。
3. 你必须从第 {start_line} 行开始继续润色，并且仅输出第 {start_line} 行到第 {line_count} 行。
4. 若你需要参考前文，请只参考“已完成润色参考”区块，不要要求重新提供前 {start_line - 1} 行原文。
""".strip()
    else:
        resume_notice = """
### Resume Notice
1. 当前没有已完成前缀，请从第 1 行开始润色。
2. 你必须从第 1 行连续输出到最后一行，不能跳号。
""".strip()

    prompt = f'''
  ## Role
  你是资深中文字幕本地化润色专家，熟悉中英双语语义和中文日常表达习惯。

  ## Task
  给定“全文{src_language}原文分行”和“全文{TARGET_LANGUAGE}草译分行”，请对{TARGET_LANGUAGE}草译做一次全文统一润色。
  目标是提升自然度、连贯性、可读性与术语一致性，同时严格保持逐行对齐。
  本次任务需要从第 {start_line} 行开始继续处理，共需产出第 {start_line} 行到第 {line_count} 行。

  ### Hard Constraints
  1. 必须从全文角度统一把握术语、文风和上下文，但本次输出只能覆盖第 {start_line} 行到第 {line_count} 行。
  2. 严禁合并、拆分、重排、增删行。
  3. 每一行输出必须是单行文本，禁止在行内再换行。
  4. 对应原文非空的行，译文不得为空。
  5. 不允许改变原文事实、立场、褒贬色彩，不允许过度引申或过度翻译。
  6. 专有名词（人名、地名、机构名、产品名等）译法必须全文统一。
  7. 同一术语和同类表达必须全文用词统一。
  8. 不要对{src_language}原文做任何改写或纠错；本任务只润色{TARGET_LANGUAGE}译文。
  9. 输出必须严格按行号顺序连续给出，不能跳号。
  10. 每一行输出格式必须严格为：`[行号] 润色结果`
  11. 必须使用半角方括号和阿拉伯数字行号，例如 `[37] 文本`
  12. 禁止输出 JSON、代码块、解释、标题、前言、后记或任何额外文本。

  ### Style and Quality Rules
  1. 允许按中文习惯调整语序（含倒装、局部换位、跨短句重组），但不得破坏行对齐约束。
  2. 优先保证“信、达、雅”：忠实原意、表达通顺、风格自然。
3. 可适度补充中文语气连接（如“则/那/故/竟”等）以增强连贯性，但不得凭空添加信息。
4. 对英文口语连接词（如 but/so）若仅为口头衔接，不要机械译成“但/所以/然而”。
5. 可适度使用地道中文表达（含成语）增强本地化，但避免生硬堆砌。
6. 字幕应尽量简洁，单行长度以“易读”为优先，理想约 15 字左右（软约束，不得因压缩而丢信息）。
7. 标点规则：除问号、感叹号、引号外，其余标点尽量弱化处理（必要时用空格替代），保持画面阅读简洁。

### Important notice
你是一个追求极度忠实、语意精准的字幕翻译。你的任务是传递原文的绝对信息与原本语气，绝不允许替说话人进行文饰。在翻译过程中，禁止使用原文未提及的成语、四字词语、网络梗或垂直圈子的俗语。禁止为了追求所谓的地道而凭空创造比喻或夸张表达。原文用词直白，译文就必须直白；原文是中性描述，译文绝不能带有情绪化的渲染。宁可译文直白平淡，也绝不画蛇添足。

  ### Content Summary
  {summary_prompt}

  {resume_notice}

  ## INPUT
  <completed_polish_reference>
  {polished_prefix_block if polished_prefix_block else "N/A"}
  </completed_polish_reference>

  <remaining_source_subtitles>
  {remaining_source_block}
  </remaining_source_subtitles>

  <remaining_draft_translation>
  {remaining_draft_block}
  </remaining_draft_translation>

  ## Output Format
  仅输出正文，不要输出任何解释或额外文本。
  从第 {start_line} 行开始逐行输出，每一行必须单独占一行，格式如下：
  [{start_line}] 第{start_line}行润色结果
  [{min(start_line + 1, line_count)}] 第{min(start_line + 1, line_count)}行润色结果
  ...
  [{line_count}] 第{line_count}行润色结果

  再强调一次：
  1. 只输出第 {start_line} 行到第 {line_count} 行
  2. 不要输出第 1 行到第 {start_line - 1} 行
  3. 不要使用 JSON
  4. 不要使用代码块
  5. 不要输出任何说明文字
  '''.strip()

    return prompt


## ================================================================
# @ step6_splitforsub.py
def get_target_semantic_alignment_prompt(alignment_items):
    source_language = load_key("whisper.detected_language")
    target_language = load_key("target_language")
    items_json = json.dumps(alignment_items, ensure_ascii=False, indent=2)

    return f'''
## 角色
你是一名专业的双语字幕语义映射专家，熟悉 {source_language} 与 {target_language} 的语序差异、语义结构和字幕阅读习惯。

## 任务
给定一批字幕映射项目。每个项目包含：

- `id`：项目编号。
- `source`：完整原文。
- `source_parts`：已经完成拆分、校验和原文映射的最终原文片段，将直接用于最终字幕时间段。
- `target`：完整译文。
- `target_boundary_candidates`：程序根据中文空格和标点提取的候选语义单元。这些内容只作为判断参考，候选边界可能来自正常语义停顿，也可能来自英文术语、数字或排版空格。

`source_parts` 是已经确定的最终原文片段。不得重新拆分、合并、改写或调整 `source_parts`。你的唯一任务是为这些固定时间片段分配译文显示文本。

请为每个 `source_parts` 元素分配一条应在对应时间段显示的译文，返回 `target_parts`。

`target_parts` 的数量必须与 `source_parts` 完全相同。

## 核心原则

1. 默认保持完整译文。当 `target` 表达的是一个连续、完整、难以自然拆开的语义单元时，将完整 `target` 复制到每个 `source_parts` 对应位置，使观众能够在多个连续时间段内阅读同一句译文。

2. 当 `target` 明确包含多个相对独立、自然完整的短句或分句，并且这些语义单元能够按照时间顺序对应 `source_parts` 时，将它们分配到相应位置。

3. 中文语义单元数量无需与原文片段数量相同：
   - 一个中文语义单元可以覆盖多个连续的原文片段，此时在这些位置重复该中文语义单元。
   - 多个相邻中文语义单元可以合并后对应一个原文片段。
   - 所有映射必须保持原文和译文的语义顺序。

4. 只在存在明确、自然的中文语义边界时使用不同的 `target_parts`。不得为了凑齐数量而强行拆分完整短语或句子。

5. `target_boundary_candidates` 仅提供候选边界：
   - 中文与英文术语之间的空格通常不构成语义边界。
   - 英文词组内部的空格不构成中文语义边界。
   - 产品名称、人物名称、机构名称、型号和数字不得从中间拆开。
   - 候选单元缺乏独立语义时，应合并或保持完整译文。

6. 每个输出部分都必须自然、完整、易于单独阅读。避免产生只包含助词、介词、连词或残缺修饰语的片段。

7. 允许自然短句以“并、但、而、所以、然后、同时、接着、还、又”等连接成分开头，只要该短句本身具有完整语义。

8. 对应关系不明确、中文语序与原文片段难以自然对应、任何拆分都会损害表达时，将完整 `target` 复制到所有对应位置。

## 译文保真约束（严格）

1. 只能使用 `target` 中已有的内容，不得重新翻译。
2. 不得改写、润色、扩写、缩写或补充信息。
3. 不得删除有效信息，不得改变事实、语气和表达含义。
4. 不得调整中文语义单元的顺序。
5. 不得改变专有名词、英文术语、型号、数字和单位。
6. 不得创造原译文中不存在的代词、连接词或解释。
7. 输出不同语义单元时，只能在 `target_boundary_candidates` 给出的自然边界处分配或合并。
8. 候选片段外缘的空格可以去除，其他字符应保持原样。
9. 当同一个中文语义单元覆盖多个连续时间段时，必须逐项返回完全相同的文本。

## 映射示例

### 示例一：三个英文片段对应三个自然中文短句

输入：

{{
  "id": 1,
  "source": "taking the sheets off of an un-camouflaged version for the full design reveal, shot that video.",
  "source_parts": [
    "taking the sheets off of an un-camouflaged version",
    "for the full design reveal,",
    "shot that video."
  ],
  "target": "就是为无伪装实车掀开幕布 做完整的设计揭晓 并拍了那支视频",
  "target_boundary_candidates": [
    "就是为无伪装实车掀开幕布",
    "做完整的设计揭晓",
    "并拍了那支视频"
  ]
}}

输出：

{{
  "id": 1,
  "target_parts": [
    "就是为无伪装实车掀开幕布",
    "做完整的设计揭晓",
    "并拍了那支视频"
  ]
}}

### 示例二：三个英文片段共同对应一个完整中文语义单元

输入：

{{
  "id": 2,
  "source": "This is what we need to get the job done properly.",
  "source_parts": [
    "This is what we need",
    "to get the job",
    "done properly."
  ],
  "target": "这样才能把这件事彻底处理好",
  "target_boundary_candidates": []
}}

输出：

{{
  "id": 2,
  "target_parts": [
    "这样才能把这件事彻底处理好",
    "这样才能把这件事彻底处理好",
    "这样才能把这件事彻底处理好"
  ]
}}

### 示例三：三个英文片段对应两个中文语义单元

输入：

{{
  "id": 3,
  "source": "We need to inspect the data and submit the report.",
  "source_parts": [
    "We need",
    "to inspect the data,",
    "and submit the report."
  ],
  "target": "我们需要检查数据 然后提交报告",
  "target_boundary_candidates": [
    "我们需要检查数据",
    "然后提交报告"
  ]
}}

输出：

{{
  "id": 3,
  "target_parts": [
    "我们需要检查数据",
    "我们需要检查数据",
    "然后提交报告"
  ]
}}

## 输入

<alignment_items>
{items_json}
</alignment_items>

## 输出要求

仅输出一个可解析的 JSON 对象，不使用 Markdown 代码块，不添加分析、理由、解释或其他文字。

输出格式：

{{
  "items": [
    {{
      "id": 1,
      "target_parts": [
        "对应第一个原文片段的译文",
        "对应第二个原文片段的译文"
      ]
    }}
  ]
}}

严格要求：

1. 每个输入项目必须对应一个输出项目。
2. `id` 必须与输入完全一致。
3. 输出项目顺序必须与输入顺序一致。
4. 每个 `target_parts` 的数量必须等于对应 `source_parts` 的数量。
5. `target_parts` 中的每一项必须是非空字符串。
6. JSON 对象只包含 `items`；每个项目只包含 `id` 和 `target_parts`。
'''.strip()


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
