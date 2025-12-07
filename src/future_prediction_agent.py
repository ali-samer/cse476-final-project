import re
from src.remote_llm import call_model_chat_completions


FUTURE_PREDICTION_SYSTEM = """
You are an agent that predicts future events.

Strict Rules you must abide by:
 - You must make a prediction. Do not refuse.
 - The final answer must end with exactly one of these forms:

       \\boxed{Yes}
       \\boxed{No}
       \\boxed{<your numeric prediction>}

 - No extra text after the box.
 - No explanations. No reasoning. Just one final line containing the box.
""".strip()

def extract_box_content(text):

    if not text:
        return None

    m = re.search(r"\\boxed\{(.+?)}", text)
    if not m:
        return None

    return m.group(1).strip()


def run_future_prediction_agent(question, verbose=False):

    r = call_model_chat_completions(
        prompt=question,
        system=FUTURE_PREDICTION_SYSTEM,
        temperature=0.0,
    )

    if not r["ok"]:
        raise RuntimeError("An API error occurred in run_future_prediction_agent: %s" % r["error"])

    raw = (r["text"] or "").strip()
    if verbose:
        print("RAW MODEL OUTPUT:")
        print(raw)
        print("-" * 40)

    content = extract_box_content(raw)
    if content is None:
        return "['No']"

    try:
        float(content)
        return f"[{content}]"
    except ValueError:
        return f"['{content}']"