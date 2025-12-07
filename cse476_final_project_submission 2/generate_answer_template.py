#!/usr/bin/env python3
"""
Generate a placeholder answer file that matches the expected auto-grader format.

Reads the input questions from cse_476_final_project_test_data.json and writes
an answers JSON file where each entry contains a string under the "output" key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from tqdm import tqdm
from src.agent import run_agent

INPUT_PATH = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def build_answers(questions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    answers: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        for idx, question in enumerate(tqdm(questions, desc="Generating answers"), start=1):
            future = executor.submit(run_agent, question["input"], domain=None, verbose=False)
            try:
                real_answer = future.result(timeout=5)  # five second timeout in-case the model takes too long to answer
                if not isinstance(real_answer, str):
                    real_answer = str(real_answer)
            except TimeoutError:
                # using an empty string in case the model does not answer
                real_answer = ""

            answers.append({"output": real_answer})

            if idx % 5 == 0:
                with OUTPUT_PATH.open("w") as fp:
                    json.dump(answers, fp, ensure_ascii=False, indent=2)

    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars). Please make sure your answer does not include any intermediate results."
            )


def main() -> None:
    questions = load_questions(INPUT_PATH)
    answers = build_answers(questions)

    with OUTPUT_PATH.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)

    with OUTPUT_PATH.open("r") as fp:
        saved_answers = json.load(fp)
    validate_results(questions, saved_answers)
    print(
        f"Wrote {len(answers)} answers to {OUTPUT_PATH} "
        "and validated format successfully."
    )


if __name__ == "__main__":
    main()