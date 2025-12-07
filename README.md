# CSE 476 Final Project
This repo contains my implementation of a simple inference-time agent for the CSE 476 final project. The agent can answer questions from different domains like math, coding, common sense, future prediction, and planning. It first figures out the domain (if not given), then runs the right domain agent, and then applies the inference techniques to clean up the answer.

The math agent is the only one that needs a tool, so I used a SymPy-based calculator for that. Everything else just uses the LLM responses directly.

## Requirements
Install these dependencies if you don't already have them:
```python
pip install sympy tqdm
```
(Those are basically the only ones you really need besides standard libs.)

## Running the Agent

To evaluate on the development dataset:
```python
python run.py
```

To generate predictions for the hidden test set, cs into the cse476_final_project_submission 2 folder and run the following script:

```python
python generate_answer_template.py
```

This will write all answers into cse_476_final_project_answers.json
and save progress every few questions.