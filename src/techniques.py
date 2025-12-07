from collections import Counter
from src.remote_llm import call_model_chat_completions


def run_with_self_consistency(sample_fn, num_samples=3, aggregate_fn=None, verbose=False):
    samples = []

    for i in range(num_samples):
        answer = sample_fn(i)
        samples.append(answer)
        if verbose:
            print("sample", i, ":", answer)

    if not samples:
        return None, []

    if aggregate_fn is not None:
        final_answer = aggregate_fn(samples)
        if verbose:
            print("final answer (custom):", final_answer)
        return final_answer, samples

    counts = Counter(samples)
    # just pick the most common answer
    final_answer = counts.most_common(1)[0][0]

    if verbose:
        print("final answer (majority vote):", final_answer)

    return final_answer, samples


def llm_judge_and_refine_generic(question, candidate, base_solver_fn, domain_name="task", verbose=False):
    debug = {
        "raw_judge_text": None,
        "refine_ran": False,
    }

    system = "You are a strict grader for " + domain_name + " problems."
    prompt = (
        "You will be given a problem and a proposed answer.\n\n"
        "Problem:\n"
        + question
        + "\n\nProposed answer:\n"
        + str(candidate)
        + "\n\nDecide if the proposed answer is likely correct and in the correct format "
          "for this kind of problem.\n\n"
          "Reply with EXACTLY one word:\n"
          "  ACCEPT\n"
          "  REJECT\n"
          "Do not say anything else."
    )

    res = call_model_chat_completions(
        prompt=prompt,
        system=system,
        temperature=0.0,
    )

    if not res["ok"]:
        if verbose:
            print("judge api error:", res.get("error"))
        return candidate, True, debug

    raw_judge_text = (res["text"] or "").strip()
    debug["raw_judge_text"] = raw_judge_text

    if verbose:
        print("judge output:", raw_judge_text)

    first_word = raw_judge_text.split()[0].lower() if raw_judge_text else ""

    if first_word.startswith("accept"):
        accepted = True
    elif first_word.startswith("reject"):
        accepted = False
    else:
        accepted = True

    if accepted:
        if verbose:
            print("judge accepted the answer.")
        return candidate, True, debug

    if verbose:
        print("judge rejected the answer, trying again once...")

    debug["refine_ran"] = True
    new_candidate = base_solver_fn()

    if verbose:
        print("refined answer:", new_candidate)

    return new_candidate, False, debug


def apply_techniques(
    question,
    base_agent_fn,
    domain_name="task",
    num_sc_samples=1,
    use_judge=True,
    verbose=False,
    aggregate_fn=None,
):
    if num_sc_samples > 1:
        def sample_fn(i):
            return base_agent_fn(question)

        candidate, samples = run_with_self_consistency(
            sample_fn=sample_fn,
            num_samples=num_sc_samples,
            aggregate_fn=aggregate_fn,
            verbose=verbose,
        )
    else:
        candidate = base_agent_fn(question)
        samples = [candidate]
        if verbose:
            print("single sample answer:", candidate)

    if not use_judge:
        return candidate

    final_candidate, accepted, debug = llm_judge_and_refine_generic(
        question=question,
        candidate=candidate,
        base_solver_fn=lambda: base_agent_fn(question),
        domain_name=domain_name,
        verbose=verbose,
    )

    return final_candidate