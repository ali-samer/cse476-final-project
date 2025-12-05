from data_utils import load_dev_data
from model_client import call_model_chat_completions

def main():
    data = load_dev_data("data/cse476_final_project_dev_data.json")
    if not data:
        print("Something went wrong. No data was found")
        return

    ex = data[0]

    print("Domain:", ex["domain"])
    print("Input:", ex["input"])
    print()

    out = call_model_chat_completions(ex["input"])
    print("Model says:")
    print(out["text"])

if __name__ == "__main__":
    main()