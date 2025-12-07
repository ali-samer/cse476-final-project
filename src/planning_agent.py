from src.remote_llm import call_model_chat_completions


PLANNING_SYSTEM_PROMPT = """
You are a planning assistant.

You will be given:
  - A description of available actions and their preconditions / effects.
  - One or more [STATEMENT] blocks that describe initial conditions and a goal.
  - Example [PLAN] ... [PLAN END] pairs that show the desired plan format.

Your task is to produce a valid plan for the final [STATEMENT] block.

Rules for your output:
  - Only output the plan for the final [STATEMENT].
  - Each action must be on a separate line.
  - Use the same parenthesized action syntax as in the examples:
        (feast b d)
        (succumb b)
        (drive truck1 depot0 distributor0)
  - Do not include [PLAN], [PLAN END], [STATEMENT], or any explanations.
  - Do not add extra text before or after the actions.
Just return the sequence of actions, one per line.
""".strip()


def _extract_plan_lines(text):
    if not text:
        return ""

    lines = [line.rstrip() for line in text.splitlines()]
    plan_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("(") and ")" in stripped:
            plan_lines.append(stripped)

    if not plan_lines:
        return text.strip() + ("\n" if text and not text.endswith("\n") else "")

    return "\n".join(plan_lines) + "\n"


def run_planning_agent(question, verbose=False):
    r = call_model_chat_completions(
        prompt=question,
        system=PLANNING_SYSTEM_PROMPT,
        temperature=0.0,
    )

    if not r["ok"]:
        raise RuntimeError("An API error occurred in run_planning_agent: %s" % r["error"])

    raw = (r["text"] or "").strip()
    if verbose:
        print("RAW PLANNING OUTPUT:")
        print(raw)
        print("-" * 40)

    plan = _extract_plan_lines(raw)

    if verbose:
        print("CLEANED PLAN:")
        print(plan)
        print("-" * 40)

    return plan