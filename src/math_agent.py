from src.remote_llm import call_with_template, MATH_SYSTEM_PROMPT, MATH_AGENT_FIRST_PROMPT, MATH_AGENT_FOLLOWUP_PROMPT
from src.utils import parse_action, parse_final_answer, sympy_calculate

def run_math_agent(question, max_tool_uses=30, verbose=False):
    r1 = call_with_template(
        MATH_AGENT_FIRST_PROMPT,
        system=MATH_SYSTEM_PROMPT,
        question=question,
    )
    if not r1["ok"]:
        raise RuntimeError("An API error occurred in run_math_agent: %s" % r1["error"])

    if verbose:
        print("LLM ->", r1["text"])

    action, payload = parse_action(r1["text"])

    while action == "CALCULATE":
        exact, numeric = sympy_calculate(payload)
        if verbose:
            print("CALC =", exact, "(≈", numeric, ")")

        rN = call_with_template(
            MATH_AGENT_FOLLOWUP_PROMPT,
            system=MATH_SYSTEM_PROMPT,
            question=question,
            calc_result=str(numeric),
        )
        if not rN["ok"]:
            raise RuntimeError("API error: %s" % rN["error"])

        if verbose:
            print("LLM →", rN["text"])

        action, payload = parse_action(rN["text"])

    final_answer, debug = parse_final_answer(payload)
    if verbose:
        print("DEBUG FINAL:", debug)
    return final_answer