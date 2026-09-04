# Benchmark validity

A task containing multiple agents, hidden types, or long histories does not automatically require opponent modeling, adaptation, or meaningful strategic memory. Before model inference, candidate environments were screened with model-independent gates:

- the best constant policy must be substantially worse than a conditioned policy;
- state/history conditioning must add measurable decision value;
- hidden type must change optimal actions;
- opponent reactions must change long-horizon decisions;
- action-value margins must be large enough to interpret;
- replay and evaluator computations must match exact references.

Three successive custom candidate environments were rejected before any LLM inference. Each implemented real interaction and passed substantial exactness checks, but each failed central competence gates: constant shortcuts remained too strong, useful history was too weak, type identification was poorly calibrated, or reactions did not alter long-horizon choices enough. The prototypes are not included here; the rejection logic is the reusable result.

Pricing is retained as a controlled diagnostic substrate, not a final proof of opponent modeling. It was later shown to admit a strong constant-policy shortcut. This limitation changes the interpretation of high Pricing score, while leaving exact within-environment measurements valid.
