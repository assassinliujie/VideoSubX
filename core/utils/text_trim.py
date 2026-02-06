import re
from rich import print as rprint
from rich.panel import Panel
from core.prompts import get_subtitle_trim_prompt
from core.utils.estimate_duration import init_estimator, estimate_duration
from core.utils import ask_gpt, load_key

ESTIMATOR = None

# 默认速度因子，用于估算阅读时长
DEFAULT_SPEED_FACTOR_MAX = 1.4

def check_len_then_trim(text, duration):
    """检查文本时长，如果过长则用 LLM 裁剪"""
    global ESTIMATOR
    if ESTIMATOR is None:
        ESTIMATOR = init_estimator()
    
    # 尝试从配置加载 speed_factor，如果不存在则使用默认值
    try:
        speed_factor = load_key("speed_factor")
        speed_max = speed_factor.get('max', DEFAULT_SPEED_FACTOR_MAX)
    except:
        speed_max = DEFAULT_SPEED_FACTOR_MAX
    
    estimated_duration = estimate_duration(text, ESTIMATOR) / speed_max
    
    rprint(f"Subtitle text: {text}, "
           f"[bold green]Estimated reading duration: {estimated_duration:.2f} seconds[/bold green]")

    if estimated_duration > duration:
        rprint(Panel(f"Estimated reading duration {estimated_duration:.2f} seconds exceeds given duration {duration:.2f} seconds, shortening...", title="Processing", border_style="yellow"))
        original_text = text
        prompt = get_subtitle_trim_prompt(text, duration)
        def valid_trim(response):
            if 'result' not in response:
                return {'status': 'error', 'message': 'No result in response'}
            return {'status': 'success', 'message': ''}
        try:    
            response = ask_gpt(prompt, resp_type='json', log_title='sub_trim', valid_def=valid_trim)
            shortened_text = response['result']
        except Exception:
            rprint("[bold red]🚫 AI refused to answer due to sensitivity, so manually remove punctuation[/bold red]")
            shortened_text = re.sub(r'[,.!?;:，。！？；：]', ' ', text).strip()
        rprint(Panel(f"Subtitle before shortening: {original_text}\nSubtitle after shortening: {shortened_text}", title="Subtitle Shortening Result", border_style="green"))
        return shortened_text
    else:
        return text
