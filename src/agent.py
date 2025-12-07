from src.utils import parse_domain_line, KNOWN_DOMAINS
from src.math_agent import run_math_agent
from src.coding_agent import run_coding_agent
from src.future_prediction_agent import run_future_prediction_agent
from src.common_sense_agent import run_common_sense_agent
from src.planning_agent import run_planning_agent
from src.universal_agent import run_universal_agent
from src.remote_llm import call_model_chat_completions

def classify_domain(question):
    system = "You are a classifier. Read the question and choose its domain."
    prompt = (
        "Choose exactly one domain for the following input.\n\n"
        "Allowed domains:\n"
        "  - math\n"
        "  - coding\n"
        "  - common_sense\n"
        "  - future_prediction\n"
        "  - planning\n\n"
        "Input:\n"
        f"{question}\n\n"
        "Reply with exactly one line of the form:\n"
        "DOMAIN: <one of the domains above>"
    )

    r = call_model_chat_completions(prompt, system=system, temperature=0.0)
    if not r["ok"]:
        raise RuntimeError("Domain classification error: %s" % r["error"])

    return parse_domain_line(r["text"])

def run_agent(question, domain=None, verbose=False):
    if domain is None:
        domain = classify_domain(question)

    if domain == "math":
        return run_math_agent(question, verbose=verbose)
    elif domain == "coding":
        return run_coding_agent(question, verbose=verbose)
    elif domain == "future_prediction":
        return run_future_prediction_agent(question, verbose=verbose)
    elif domain == "common_sense":
        return run_common_sense_agent(question, verbose=verbose)
    elif domain == "planning":
        return run_planning_agent(question, verbose=verbose)
    else:
        return run_universal_agent(question, verbose=verbose)