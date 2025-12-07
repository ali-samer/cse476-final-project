from src.remote_llm import call_model_chat_completions


UNIVERSAL_FALLBACK_SYSTEM = """
You are a helpful assistant.
Answer the user's prompt directly.
Do not say anything extra.
""".strip()


def run_universal_agent(question, verbose=False):
    r = call_model_chat_completions(
        prompt=question,
        system=UNIVERSAL_FALLBACK_SYSTEM,
        temperature=0.0,
    )

    if not r["ok"]:
        raise RuntimeError("An API error occurred in run_universal_agent: %s" % r["error"])

    text = (r["text"] or "").strip()

    if verbose:
        print("UNIVERSAL RAW OUTPUT:")
        print(text)
        print("-" * 40)

    return text