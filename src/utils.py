from sympy import (
    sympify, sqrt, pi, E,
    sin, cos, tan,
    asin, acos, atan,
    log, exp
)
from sympy.core.sympify import SympifyError
import re
import json

KNOWN_DOMAINS = [
    "math",
    "coding",
    "common_sense",
    "future_prediction",
    "planning",
]

SAFE_NAMESPACE = {
    "sqrt": sqrt,
    "pi": pi,
    "e": E,
    "E": E,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "asin": asin,
    "acos": acos,
    "atan": atan,
    "log": log,   # natural log
    "ln": log,    # alias
    "exp": exp,
}

def load_dev_data(path):
    with open(path, "r") as f:
        raw = json.load(f)

    items = []
    for row in raw:
        items.append({
            "domain": row["domain"],
            "input": row["input"],
            "expected_output": row["output"],
        })
    return items

def sympy_calculate(expr_str):
    if not isinstance(expr_str, str):
        raise TypeError("Expression must be a string")

    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty expression")

    try:
        expr = sympify(expr_str, locals=SAFE_NAMESPACE, evaluate=True)
    except SympifyError as e:
        raise ValueError("Invalid expression: %r" % expr_str) from e

    exact_value = expr.simplify()
    numeric_value = exact_value.evalf()
    return exact_value, numeric_value

def parse_action(raw_text):
    if not raw_text:
        return "OTHER", ""

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "OTHER", ""

    first = lines[0]

    upper = first.upper()
    if upper.startswith("CALCULATE:"):
        payload = first.split(":", 1)[1].strip()
        return "CALCULATE", payload

    if upper.startswith("FINAL:"):
        payload = first.split(":", 1)[1].strip()
        return "FINAL", payload

    return "FINAL", first


def extract_first_integer(text):
    if not text:
        return None

    m = re.search(r"-?\d+", text)
    if m:
        return m.group(0)
    return None


def extract_last_integer(text):
    if not text:
        return None

    nums = re.findall(r"-?\d+", text)
    if not nums:
        return None
    return nums[-1]


def parse_final_answer(raw_text):
    action, payload = parse_action(raw_text)

    if action != "FINAL":
        return payload, {"action": action, "payload": payload, "parsed_int": None}

    ans = extract_last_integer(payload)
    if ans is None:
        ans = extract_first_integer(payload)

    if ans is None:
        return payload, {"action": action, "payload": payload, "parsed_int": None}

    return ans, {"action": action, "payload": payload, "parsed_int": ans}

def parse_domain_line(raw_text):
    if not raw_text:
        return None

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return None

    first = lines[0]

    m = re.search(r"domain\s*[:=\-]\s*([A-Za-z_]+)", first, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().lower()
        if candidate in KNOWN_DOMAINS:
            return candidate

    candidate = first.strip().lower()
    if candidate in KNOWN_DOMAINS:
        return candidate

    return None