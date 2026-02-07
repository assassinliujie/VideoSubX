import os
import json
import time
from threading import Lock

import json_repair
import requests
from openai import OpenAI
from rich import print as rprint

from core.utils.config_utils import load_key

# ------------
# cache gpt response
# ------------

LOCK = Lock()
GPT_LOG_FOLDER = "output/gpt_log"


def _load_key_or_default(key, default):
    try:
        return load_key(key)
    except Exception:
        return default


def _to_int(value, default, min_value=None):
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    return parsed


def _to_float(value, default, min_value=None):
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if min_value is not None:
        parsed = max(min_value, parsed)
    return parsed


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _pick_setting(override: dict, key: str, default=None):
    if isinstance(override, dict):
        value = override.get(key, None)
        if value not in (None, ""):
            return value
    return default


def _save_cache(model, prompt, resp_content, resp_type, resp, message=None, log_title="default"):
    with LOCK:
        logs = []
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(
            {
                "model": model,
                "prompt": prompt,
                "resp_content": resp_content,
                "resp_type": resp_type,
                "resp": resp,
                "message": message,
            }
        )
        with open(file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)


def _load_cache(prompt, resp_type, log_title, model=None):
    with LOCK:
        file = os.path.join(GPT_LOG_FOLDER, f"{log_title}.json")
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if item["prompt"] == prompt and item["resp_type"] == resp_type:
                        if model is not None and item.get("model") != model:
                            continue
                        return item["resp"]
        return False


def _is_claude_model(model: str) -> bool:
    return isinstance(model, str) and "claude" in model.lower()


def _normalize_openai_base_url(base_url: str) -> str:
    if "ark" in base_url:
        return "https://ark.cn-beijing.volces.com/api/v3"  # huoshan base url
    if "v1" not in base_url:
        return base_url.strip("/") + "/v1"
    return base_url


def _normalize_claude_messages_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    if base_url.endswith("/v1/messages"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/messages"
    return base_url + "/v1/messages"


def _safe_json(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _ask_gpt_chat(prompt, resp_type, model, base_url, api_key, llm_support_json, timeout):
    base_url = _normalize_openai_base_url(base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response_format = {"type": "json_object"} if resp_type == "json" and llm_support_json else None

    params = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        timeout=timeout,
    )
    resp_raw = client.chat.completions.create(**params)
    resp_content = resp_raw.choices[0].message.content
    if resp_type == "json":
        resp = json_repair.loads(resp_content)
    else:
        resp = resp_content
    return resp_content, resp


def _ask_claude_messages(prompt, resp_type, model, base_url, api_key, llm_support_json, timeout):
    url = _normalize_claude_messages_url(base_url)
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    # Native Claude structured output path: force a tool call that returns JSON object input.
    if resp_type == "json" and llm_support_json:
        payload["tools"] = [
            {
                "name": "output_json",
                "description": "Return final answer as a JSON object only.",
                "input_schema": {"type": "object", "additionalProperties": True},
            }
        ]
        payload["tool_choice"] = {"type": "tool", "name": "output_json"}

    resp_raw = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp_raw.raise_for_status()
    data = resp_raw.json()
    blocks = data.get("content", []) if isinstance(data, dict) else []

    if resp_type == "json":
        for block in blocks:
            if block.get("type") == "tool_use" and "input" in block:
                tool_input = block["input"]
                if isinstance(tool_input, dict):
                    return _safe_json(tool_input), tool_input
                return str(tool_input), json_repair.loads(str(tool_input))

        text_content = "\n".join(
            block.get("text", "") for block in blocks if block.get("type") == "text"
        ).strip()
        if not text_content:
            text_content = _safe_json(data)
        return text_content, json_repair.loads(text_content)

    text_content = "\n".join(
        block.get("text", "") for block in blocks if block.get("type") == "text"
    ).strip()
    if not text_content:
        for block in blocks:
            if block.get("type") == "tool_use" and "input" in block:
                tool_input = block["input"]
                text_content = (
                    _safe_json(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
                )
                break
    if not text_content:
        text_content = _safe_json(data)
    return text_content, text_content


# ------------
# ask gpt with retry
# ------------

def ask_gpt(prompt, resp_type=None, valid_def=None, log_title="default", api_settings=None):
    global_api_key = _load_key_or_default("api.key", "")
    global_model = _load_key_or_default("api.model", "")
    global_base_url = _load_key_or_default("api.base_url", "")
    global_llm_support_json = _load_key_or_default("api.llm_support_json", True)
    global_timeout = _load_key_or_default("api.request_timeout_sec", 300)
    global_retries = _load_key_or_default("api.request_retries", 5)
    global_retry_delay = _load_key_or_default("api.request_retry_delay_sec", 1)

    api_key = _pick_setting(api_settings, "key", global_api_key)
    if not api_key:
        raise ValueError("API key is not set")
    model = _pick_setting(api_settings, "model", global_model)

    # check cache
    cached = _load_cache(prompt, resp_type, log_title, model=model)
    if cached:
        rprint("use cache response")
        return cached

    base_url = _pick_setting(api_settings, "base_url", global_base_url)
    llm_support_json = _to_bool(
        _pick_setting(api_settings, "llm_support_json", global_llm_support_json),
        default=True,
    )
    timeout = _to_int(_pick_setting(api_settings, "request_timeout_sec", global_timeout), 300, min_value=1)
    retries = _to_int(_pick_setting(api_settings, "request_retries", global_retries), 5, min_value=0)
    retry_delay = _to_float(
        _pick_setting(api_settings, "request_retry_delay_sec", global_retry_delay),
        1,
        min_value=0,
    )
    route = "claude-messages" if _is_claude_model(model) else "openai-chat"

    total_attempts = retries + 1
    last_exception = None

    for attempt in range(1, total_attempts + 1):
        req_start = time.time()
        rprint(
            f"[cyan]LLM request start[/cyan] model={model} route={route} timeout={timeout}s "
            f"log={log_title} attempt={attempt}/{total_attempts}"
        )
        try:
            if _is_claude_model(model):
                resp_content, resp = _ask_claude_messages(
                    prompt=prompt,
                    resp_type=resp_type,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    llm_support_json=llm_support_json,
                    timeout=timeout,
                )
            else:
                resp_content, resp = _ask_gpt_chat(
                    prompt=prompt,
                    resp_type=resp_type,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    llm_support_json=llm_support_json,
                    timeout=timeout,
                )

            req_cost = round(time.time() - req_start, 2)
            rprint(
                f"[green]LLM request done[/green] model={model} route={route} cost={req_cost}s "
                f"log={log_title} attempt={attempt}/{total_attempts}"
            )

            # check if the response format is valid
            if valid_def:
                valid_resp = valid_def(resp)
                if valid_resp["status"] != "success":
                    _save_cache(
                        model,
                        prompt,
                        resp_content,
                        resp_type,
                        resp,
                        log_title="error",
                        message=valid_resp["message"],
                    )
                    raise ValueError(f"API response error: {valid_resp['message']}")

            _save_cache(model, prompt, resp_content, resp_type, resp, log_title=log_title)
            return resp
        except Exception as e:
            req_cost = round(time.time() - req_start, 2)
            last_exception = e
            rprint(
                f"[red]LLM request failed[/red] model={model} route={route} cost={req_cost}s "
                f"log={log_title} attempt={attempt}/{total_attempts} error={e}"
            )
            if attempt < total_attempts and retry_delay > 0:
                rprint(f"[yellow]Retrying in {retry_delay}s...[/yellow]")
                time.sleep(retry_delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("LLM request failed with unknown error")


if __name__ == "__main__":
    result = ask_gpt("""test respond ```json\n{\"code\": 200, \"message\": \"success\"}\n```""", resp_type="json")
    rprint(f"Test json output result: {result}")
