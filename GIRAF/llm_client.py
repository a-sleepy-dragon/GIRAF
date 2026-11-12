import json
import re
import time
import urllib.error
import urllib.request

import config

_last_usage = None

_cum_calls = 0
_cum_prompt_tokens = 0
_cum_completion_tokens = 0
_cum_wall_s = 0.0


def get_last_usage():
    return _last_usage


def reset_usage_stats():
    global _cum_calls, _cum_prompt_tokens, _cum_completion_tokens, _cum_wall_s
    _cum_calls = _cum_prompt_tokens = _cum_completion_tokens = 0
    _cum_wall_s = 0.0


def get_usage_stats():
    return {
        "calls": _cum_calls,
        "prompt_tokens": _cum_prompt_tokens,
        "completion_tokens": _cum_completion_tokens,
        "total_tokens": _cum_prompt_tokens + _cum_completion_tokens,
        "model_wall_s": round(_cum_wall_s, 3),
    }


def _call_once(messages, temperature, max_tokens):
    payload = {
        "model": config.MODEL_NAME,
        "messages": messages,
        "temperature": config.TEMPERATURE if temperature is None else temperature,
        "max_tokens": max_tokens if max_tokens is not None else config.MAX_TOKENS,
    }
    if getattr(config, "REASONING_EFFORT", None) is not None:
        payload["reasoning_effort"] = config.REASONING_EFFORT
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config.BASE_URL + "/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.API_KEY,
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=config.TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"LLM HTTP {e.code}: {detail[:500]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM network error: {e}")
    wall = time.time() - t0

    global _last_usage, _cum_calls, _cum_prompt_tokens, _cum_completion_tokens, _cum_wall_s
    _last_usage = body.get("usage")
    u = body.get("usage") or {}
    _cum_calls += 1
    _cum_prompt_tokens += int(u.get("prompt_tokens") or 0)
    _cum_completion_tokens += int(u.get("completion_tokens") or 0)
    _cum_wall_s += wall

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"LLM response missing choices: {json.dumps(body)[:500]}")


def call_llm(messages, temperature=None, max_tokens=None, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            content = _call_once(messages, temperature, max_tokens)
            if content and content.strip():
                return content
            last_err = RuntimeError("LLM returned empty content (occasional with reasoning models)")
        except Exception as e:
            last_err = e
        if attempt < retries:
            time.sleep(2 * (attempt + 1))
    raise last_err


def _strip_code_fences(text):
    text = (text or "").strip()
    m = re.match(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    return text


def extract_json(text):
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        lo = text.find(open_ch)
        hi = text.rfind(close_ch)
        if lo != -1 and hi > lo:
            try:
                return json.loads(text[lo:hi + 1])
            except Exception:
                continue
    raise ValueError(f"Unable to parse JSON from response (first 300 chars): {text[:300]}")


def call_llm_json(messages, temperature=None):
    return extract_json(call_llm(messages, temperature=temperature))


def ping():
    return call_llm(
        [{"role": "user", "content": "Reply with exactly two characters: OK"}],
        max_tokens=512,
    )
