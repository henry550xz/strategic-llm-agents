# Licensing and code provenance

## Repository license position

The source research repository had no top-level license. This public repository therefore does **not** claim an open-source grant. The root notice reserves rights while allowing the repository to be publicly inspected. Library imports are ordinary API use and do not copy library source.

## File classification

| Public files | Classification | Basis |
|---|---|---|
| `src/strategic_agents/envs/pricing.py` | refactored original project code | derived from the project's Pricing environment and core dataclasses |
| `src/strategic_agents/agents/ippo_controller.py` | refactored original project code | derived from the learned Pricing controller and feature encoder |
| `src/strategic_agents/training/ippo.py` | refactored original project code | extracted GAE and clipped-PPO computation |
| `src/strategic_agents/training/sft.py` | refactored original project code | compact form of the project's Transformers/PEFT SFT workflow |
| `src/strategic_agents/training/grpo.py` | refactored original project code | compact form of the project's TRL workflow and strict action support |
| `src/strategic_agents/evaluation/regret.py` | refactored original project code | extracted exact Pricing counterfactual evaluator |
| `src/strategic_agents/evaluation/{collapse,opponent_shift,bootstrap,history,replay}.py` | original/refactored project code | condensed from project analysis protocols and metrics |
| `src/strategic_agents/representations/*.py` | refactored original project code | extracted frozen lossless serializer logic |
| `src/strategic_agents/envs/openspiel.py` | rewritten original project code | readable rewrite of project OpenSpiel utilities using public OpenSpiel APIs |
| `src/strategic_agents/agents/{api_direct,planner,rule_controller,local_llm}.py` | original project glue | small public interfaces representing implemented architectures |
| `src/strategic_agents/data/{craigslist,deal_or_no_deal}.py` | rewritten original project code | reduced project-authored adapters; no upstream records or implementation copied |
| `scripts/*.py`, `tests/*.py` | original project glue/verification | authored for this standalone synthesis |
| `configs/*.toml`, `results/*.csv` | trivial configuration / validated factual data | configurations and transcribed canonical aggregates are not third-party code |
| `README.md`, `docs/*.md`, `figures/*.svg` | original documentation and graphics | authored for this standalone synthesis |
| `.github/workflows/ci.yml`, `.gitignore`, `pyproject.toml`, `CITATION.cff` | trivial repository metadata | conventional configuration |

No substantive file is classified as copied/derived third-party code or uncertain provenance. This classification concerns code origin, not a representation that every collaborator has granted a reuse license. Accordingly, no open-source license is asserted.

## Third-party dependencies and data

- NumPy, PyTorch, Transformers, PEFT, TRL, and OpenSpiel are dependencies invoked through documented APIs; their source is not vendored.
- CraigslistBargain data must be obtained from Stanford CoCoA under its upstream terms.
- Deal-or-No-Deal data must be obtained from the upstream End-to-End Negotiator repository and is identified there as CC BY-NC 4.0.
- No raw dataset, trained adapter, checkpoint, tokenizer, or model snapshot is redistributed.

Detailed source-path mappings and parity results are in [`source_provenance.md`](source_provenance.md).
