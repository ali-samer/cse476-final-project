from collections import Counter


def self_consistency(sample_fn, num_samples=3, aggregate_fn=None, verbose=False):
    samples = []
    for i in range(num_samples):
        candidate = sample_fn(i)
        samples.append(candidate)
        if verbose:
            print(f"[self-consistency] sample {i}: {candidate}")

    if not samples:
        return None, []

    if aggregate_fn is not None:
        final = aggregate_fn(samples)
        if verbose:
            print(f"[self-consistency] final (custom aggregate): {final}")
        return final, samples

    counter = Counter(samples)
    most_common = counter.most_common()
    max_count = most_common[0][1]
    tied_values = [val for val, count in most_common if count == max_count]

    if len(tied_values) == 1:
        final = tied_values[0]
    else:
        final = None
        for s in samples:
            if s in tied_values:
                final = s
                break

    if verbose:
        print(f"[self-consistency] final (majority vote): {final}")
    return final, samples


def judge_and_refine(
    question,
    candidate,
    judge_fn,
    parse_judge_fn,
    refine_fn=None,
    parse_refine_fn=None,
    verbose=False,
):
    debug = {
        "raw_judge_text": None,
        "raw_refined_text": None,
    }

    raw_judge_text = judge_fn(question, candidate)
    debug["raw_judge_text"] = raw_judge_text

    judge_accepted = parse_judge_fn(raw_judge_text)
    if verbose:
        print("[judge_and_refine] judge_accepted:", judge_accepted)

    if judge_accepted or refine_fn is None:
        return candidate, True, debug

    raw_refined_text = refine_fn(question, candidate, raw_judge_text)
    debug["raw_refined_text"] = raw_refined_text

    if parse_refine_fn is not None:
        refined_candidate = parse_refine_fn(raw_refined_text)
    else:
        refined_candidate = raw_refined_text

    if verbose:
        print("[judge_and_refine] refined_candidate:", refined_candidate)

    return refined_candidate, False, debug