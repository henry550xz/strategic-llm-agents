# Agent architectures

| ID | Decision path | Learning role |
|---|---|---|
| S1 | state/history → API LLM → legal action | direct policy |
| S2A | state/history → API planner → semantic intent → rule controller → action | hierarchical baseline |
| S2B | state/history → API planner → semantic intent; decentralized state + intent → shared IPPO → independently sampled per-agent action | learned MARL controller |
| S3 | state/history → SFT/GRPO local LLM → legal action | locally trained direct policy |

All language-model boundaries use strict executable actions. Formatting validity, fallback behavior, and environment legality are evaluated independently from reward quality.
