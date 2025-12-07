import random
from src.agent import run_agent
from src.utils import load_dev_data

DEV_DATA_PATH = "data/cse476_final_project_dev_data.json"

DEV_MODE = True
shuffle_data = lambda d: random.sample(d, len(d))

def main():
    data = load_dev_data(DEV_DATA_PATH)
    data = shuffle_data(data)

    examples_to_run = min(5, len(data))

    print(f"Running {examples_to_run} example(s)...")
    print("=" * 80)

    for idx in range(examples_to_run):
        ex = data[idx]
        domain = ex["domain"]
        question = ex["input"]
        expected = ex["expected_output"]

        prediction = run_agent(question, domain=domain, verbose=True)

        print(f"Example {idx}")
        print(f"Domain: {domain}")
        print(f"Input: {question}")

        if DEV_MODE:
            print(f"Expected output: {expected}")

        print(f"Prediction: {prediction}")
        print("=" * 80)


if __name__ == "__main__":
    main()