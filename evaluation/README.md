# Evaluation

The evaluation harness will measure:

- Intent accuracy and macro-F1
- Per-intent performance and confusion matrix
- Historical-resolution retrieval Recall@1 / Recall@5
- Reply correctness, groundedness, relevance, and helpfulness
- Escalation precision/recall and false-auto-handling rate
- LLM-as-judge agreement with human ratings

A held-out golden set of 150–250 hand-labelled examples will be used. Retrieval and evaluation data will be separated to reduce leakage.