from src.remote_llm import call_model_chat_completions

CODING_SYSTEM_PROMPT = """
You are a coding assistant.

You will be given a task description that already includes the following:
  - the required imports
  - the function signature (def task_func(...):) inside a code block

Your job is to only write the body of that function.

Strict Rules you must abide by:
  - DO NOT repeat the imports.
  - DO NOT repeat the function definition line.
  - DO NOT wrap your answer in ``` code fences.
  - DO NOT add explanations or comments outside the code.
Just return the indented function body lines.
""".strip()


def strip_codeblock_fences(text):
    if not text:
        return ""

    text = text.strip()

    if "```" not in text:
        return text

    parts = text.split("```")
    if len(parts) >= 3:
        inner = parts[1]
        lines = inner.splitlines()
        if lines and lines[0].strip().lower().startswith("python"):
            lines = lines[1:]
        return "\n".join(lines).strip()

    return text.replace("```", "").strip()


def run_coding_agent(question, verbose=False):
    prompt = (
        "You are given a coding task.\n\n"
        "Task description:\n"
        f"{question}\n\n"
        "Remember the rules:\n"
        "  - The code stub (imports + def task_func(...):) is already provided.\n"
        "  - You must return ONLY the indented body of task_func.\n"
        "  - Do NOT repeat the imports or the def line.\n"
        "  - Do NOT wrap the code in ``` fences.\n\n"
        "Write the function body now:"
    )

    r = call_model_chat_completions(
        prompt=prompt,
        system=CODING_SYSTEM_PROMPT,
        temperature=0.0,
    )

    if not r["ok"]:
        raise RuntimeError("An API error occurred in run_coding_agent: %s" % r["error"])

    raw = (r["text"] or "").strip()
    if verbose:
        print("RAW CODE OUTPUT:")
        print(raw)
        print("-" * 40)

    body = strip_codeblock_fences(raw)

    body_lines = body.splitlines()
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    cleaned_body = "\n".join(body_lines)

    if verbose:
        print("CLEANED FUNCTION BODY:")
        print(cleaned_body)
        print("-" * 40)

    return cleaned_body