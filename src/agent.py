from src.utils import parse_domain_line, KNOWN_DOMAINS
from src.math_agent import run_math_agent
from src.coding_agent import run_coding_agent
from src.future_prediction_agent import run_future_prediction_agent
from src.common_sense_agent import run_common_sense_agent
from src.planning_agent import run_planning_agent
from src.universal_agent import run_universal_agent
from src.techniques import *

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
        return "universal"

    domain = parse_domain_line(r["text"])
    if domain is None:
        return "universal"
    return domain


DOMAIN_TO_AGENT = {
    "math":              run_math_agent,
    "coding":            run_coding_agent,
    "future_prediction": run_future_prediction_agent,
    "common_sense":      run_common_sense_agent,
    "planning":          run_planning_agent,
}

DOMAIN_TECHNIQUE_CONFIG = {
    "math": {
        "num_sc_samples": 3,
        "use_judge": True,
    },
    "coding": {
        "num_sc_samples": 2,
        "use_judge": True,
    },
    "future_prediction": {
        "num_sc_samples": 1,
        "use_judge": False,
    },
    "common_sense": {
        "num_sc_samples": 2,
        "use_judge": False,
    },
    "planning": {
        "num_sc_samples": 1,
        "use_judge": False,
    },
}

def run_agent(question, domain=None, verbose=False):
    if domain is None:
        domain = classify_domain(question)

    if domain not in DOMAIN_TO_AGENT:
        if verbose:
            print(f"[run_agent] Unknown domain '{domain}', using universal fallback.")
        return run_universal_agent(question, verbose=verbose)

    base_agent = DOMAIN_TO_AGENT[domain]
    cfg = DOMAIN_TECHNIQUE_CONFIG.get(domain, {"num_sc_samples": 1, "use_judge": False})

    base_agent_fn = lambda q: base_agent(q, verbose=False)

    final_answer = apply_techniques(
        question=question,
        base_agent_fn=base_agent_fn,
        domain_name=domain,
        num_sc_samples=cfg.get("num_sc_samples", 1),
        use_judge=cfg.get("use_judge", False),
        verbose=verbose,
    )
    return final_answer