import collections


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute the F1 score between two strings using token-level overlap.
    Both prediction and ground_truth are normalized (lowercased and split by whitespace).
    """
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()

    # Edge case: both are empty.
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def evaluate_squad_answer(ground_truths, plausible_answers, predicted, answerable=True, f1_threshold=0.6) -> int:
    """
    Evaluate a predicted answer for SQuAD 2.0.

    For answerable questions:
      - Return 1 if the predicted answer exactly matches any ground truth answer (after normalization)
        or if the maximum F1 score (against ground truth answers) is >= f1_threshold.
      - Otherwise, return 0.

    For unanswerable questions:
      - Return 1 if the predicted answer contains keywords indicating unanswerability
        (e.g., "unanswerable", "none", or is an empty string).
      - Return 0 if the predicted answer exactly matches any plausible answer
        or if the maximum F1 score (against plausible answers) is >= f1_threshold.
      - Otherwise, return -1.
    """
    # Normalize predicted answer (strip whitespace and lowercase)
    pred_norm = predicted.strip().lower()

    if answerable:
        # Check for exact match with ground truth answers
        for gt in ground_truths:
            if pred_norm == gt.strip().lower():
                return 1

        # Compute maximum F1 over all ground truth answers
        max_f1 = max(compute_f1(pred_norm, gt.strip()) for gt in ground_truths)
        return 1 if max_f1 >= f1_threshold else 0

    else:
        # For unanswerable questions, check if the prediction signals unanswerability.
        # Here we look for keywords "unanswerable", "none", or an empty string.
        unans_keywords = ['unanswerable', 'none', "i don't know"]
        if pred_norm == "" or any(kw in pred_norm for kw in unans_keywords):
            return 1

        # Check for exact match with plausible answers
        for pa in plausible_answers:
            if pred_norm == pa.strip().lower():
                return 0

        # Compute maximum F1 over plausible answers
        max_f1 = max(compute_f1(pred_norm, pa.strip()) for pa in plausible_answers)
        if max_f1 >= f1_threshold:
            return 0

        return -1


# Example usage:
if __name__ == "__main__":
    # # Example for an answerable question
    # gt_answers = ["New York City", "NYC"]
    # plausible = ["New York", "Big Apple"]
    # predicted_ans = "New York City"
    # print("Answerable evaluation:", evaluate_squad_answer(gt_answers, plausible, predicted_ans, answerable=True))
    #
    # # Example for an unanswerable question
    # gt_answers_unans = []
    # plausible_unans = ["No answer", "Not answerable", "N/A"]
    # predicted_ans_unans = "unanswerable"
    # print("Unanswerable evaluation:",
    #       evaluate_squad_answer(gt_answers_unans, plausible_unans, predicted_ans_unans, answerable=False))