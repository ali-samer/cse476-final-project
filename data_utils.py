import json

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
