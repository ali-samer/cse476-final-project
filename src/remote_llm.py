import os
import requests

API_KEY  = os.getenv("OPENAI_API_KEY", "cse476")
API_BASE = os.getenv("API_BASE", "http://10.4.58.53:41701/v1")
MODEL    = os.getenv("MODEL_NAME", "bens_model")

def call_model_chat_completions(prompt: str,
                                system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
                                model: str = MODEL,
                                temperature: float = 0.0,
                                timeout: int = 60) -> dict:
    """
    Calls an OpenAI-style /v1/chat/completions endpoint and returns:
    { 'ok': bool, 'text': str or None, 'raw': dict or None, 'status': int,
      'error': str or None, 'headers': dict }
    """
    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        status = resp.status_code
        hdrs   = dict(resp.headers)
        if status == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"ok": True, "text": text, "raw": data, "status": status, "error": None, "headers": hdrs}
        else:
            try:
                err_text = resp.json()
            except Exception:
                err_text = resp.text
            return {"ok": False, "text": None, "raw": None, "status": status, "error": str(err_text), "headers": hdrs}
    except requests.RequestException as e:
        return {"ok": False, "text": None, "raw": None, "status": -1, "error": str(e), "headers": {}}


def render_template(template, **kwargs):
    """
    Render a prompt template using str.format(**kwargs).

    Example:
        template = "Solve the problem: {question}\nUse the value x = {x}."
        prompt = render_template(template, question=q, x=3)
    """
    return template.format(**kwargs)


def call_with_template(template,
                       system,
                       temperature=0.0,
                       model=MODEL,
                       timeout=60,
                       **kwargs):
    """
    Convenience helper: fill a template and immediately call the LLM.

    Example:
        r = call_with_template(
                MATH_AGENT_FIRST_PROMPT,
                system=MATH_SYSTEM_PROMPT,
                question=my_question
            )
    """
    prompt = render_template(template, **kwargs)
    return call_model_chat_completions(
        prompt=prompt,
        system=system,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )


MATH_SYSTEM_PROMPT = """
You are a math assistant that can use a calculator tool.

You MUST respond with exactly one line in one of these forms:

1) CALCULATE: <arithmetic expression>
   - Use this when you need help evaluating a numeric expression.
   - The expression must be valid Python-style math using only:
       numbers, +, -, *, /, **, parentheses, sqrt(), pi, e, sin(), cos(), tan().

2) FINAL: <integer answer>
   - Use this when you are ready to give the final numeric answer to the problem.

Do not explain your reasoning in the output. Just return CALCULATE or FINAL.
""".strip()


MATH_AGENT_FIRST_PROMPT = """
You will solve a contest-style math problem.

Problem:
{question}

Think about the steps in your head.
If you need arithmetic, respond with:
CALCULATE: <expression>

When you know the final answer, respond with:
FINAL: <integer>

Respond with exactly one line.
""".strip()


MATH_AGENT_FOLLOWUP_PROMPT = """
You are continuing to solve the same problem.

Original problem:
{question}

The result of your previous CALCULATE expression is:
{calc_result}

Now, either:
  - If you need more arithmetic, respond with: CALCULATE: <expression>
  - If you know the final answer, respond with: FINAL: <integer>

Respond with exactly one line.
""".strip()