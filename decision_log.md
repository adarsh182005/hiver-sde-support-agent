# Decision Log

This file will record 10–15 non-obvious engineering decisions made during the assignment.

Each entry will capture:

- Decision
- Why it was made
- Alternative considered
- Trade-off

Initial decisions:

1. Keep the raw Twitter dataset out of Git because it is large and unnecessary for reproduction.
2. Keep the implementation focused on one brand, as required by the assignment, rather than building a multi-brand system.
3. Treat evaluation as a first-class component rather than optimizing only the demo path.
4. Use a held-out golden set so evaluation examples do not become retrieval evidence.
5. Compare against both a trivial and a classical ML baseline before claiming an LLM improvement.
