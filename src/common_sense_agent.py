from src.remote_llm import call_model_chat_completions

COMMON_SENSE_SYSTEM = """
You are a knowledgeable assistant that answers factual, common-sense questions.

Strict Rules you must abide by:
  - Answer with a single short phrase (a name, place, or object).
  - Do NOT give explanations.
  - Do NOT add extra sentences.
  - Do NOT add quotes or Markdown formatting.
Just return the answer phrase.
""".strip()

def clean_common_sense_answer(text):

    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    ans = lines[0]

    ans = ans.strip()
    if (ans.startswith('"') and ans.endswith('"')) or \
       (ans.startswith("'") and ans.endswith("'")):
        ans = ans[1:-1].strip()

    if ans.endswith("."):
        ans = ans[:-1].strip()

    return ans


def run_common_sense_agent(question, verbose=False):
    r = call_model_chat_completions(
        prompt=question,
        system=COMMON_SENSE_SYSTEM,
        temperature=0.0,
    )

    if not r["ok"]:
        raise RuntimeError("An API error occurred in run_common_sense_agent: %s" % r["error"])

    raw = (r["text"] or "").strip()
    if verbose:
        print("RAW COMMON_SENSE OUTPUT:")
        print(raw)
        print("-" * 40)

    answer = clean_common_sense_answer(raw)

    if verbose:
        print("CLEANED ANSWER:", answer)
        print("-" * 40)

    return answer