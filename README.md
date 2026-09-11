# Hiver SDE Intern — AI Support Agent

Take-home assignment: build an AI support agent for one brand from the Customer Support on Twitter dataset.

## Current status

This repository is the initial project scaffold. The implementation will be built in stages so that every major decision can be evaluated and reproduced.

## Planned pipeline

1. Inspect and sample the Twitter support dataset.
2. Select one brand using data-driven criteria.
3. Reconstruct usable customer-support conversations.
4. Define a small intent taxonomy from the selected brand's data.
5. Build a 150–250 example human-labelled golden evaluation set.
6. Implement two baselines: majority class and TF-IDF classifier/retrieval.
7. Implement the support agent: intent classification, historical-resolution retrieval, reply drafting, and escalation.
8. Evaluate intent, retrieval, reply quality, and escalation quality.
9. Validate the LLM judge against human ratings.
10. Document failure modes, misleading headline metrics, and engineering decisions.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── decision_log.md
├── data/
│   └── README.md
├── src/
│   └── README.md
├── baselines/
│   └── README.md
├── evaluation/
│   └── README.md
└── report/
    └── README.md
```

## Reproduction target

The final README will document a runnable path that reproduces the headline evaluation results in under 15 minutes on a prepared subsample. Full-dataset execution is intentionally out of scope.

## Data

The primary source is the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`). Raw dataset files are not committed to this repository.

## Scope

The system will focus on classification, historically grounded reply drafting, escalation, and rigorous evaluation. Production Twitter integration, real-time CRM actions, fine-tuning, and multi-brand orchestration are out of scope for this take-home.
