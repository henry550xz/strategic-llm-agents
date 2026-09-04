# Source provenance and parity

The public repository is a presentation/synthesis artifact. It does not change the classifications of source experiments.

| Public module | Original source | Treatment | Parity check |
|---|---|---|---|
| `envs/pricing.py` | `src/mas_benchmark/envs/pricing.py`, `core/types.py` | refactored, type layer inlined | frozen action sequence: observations/rewards compared exactly |
| `evaluation/regret.py` | `analysis/pricing_counterfactual.py` | refactored exact observed-share path | frozen cases and environment transitions compared within `1e-12` |
| `agents/ippo_controller.py` | `controllers/learned_pricing_controller.py` | refactored network/features | fixed observation feature vector compared exactly; parameter shapes checked |
| `training/ippo.py` | `training/pricing_ippo.py::collect_ippo_rollout`, `_finish_episode_gae`, `ppo_update` | extracted rollout and optimization core | frozen GAE vector matched; shared rollout invariants and PPO finite-gradient behavior tested |
| `training/sft.py` | `training/train_sft.py` | compact executable LoRA/Trainer workflow | GPU training not rerun publicly; trained-run evidence remains in source artifacts |
| `training/grpo.py` | `training/train_grpo.py`, `grpo_action_support.py` | compact executable TRL workflow with strict support | strict grid/off-grid parser tested; GPU training evidence remains in source artifacts |
| `representations/*` | `scripts/build_context_format_protocol.py` | extracted compact/verbose/redundant transforms | round-trip canonical equality |
| `evaluation/history.py` | history diagnostic builders/scripts | compact reusable intervention | invariance/removal/shuffle unit tests |
| `envs/openspiel.py` | `src/mas_benchmark/drh_openspiel.py` | rewritten for readability | optional Kuhn enumeration/CFR+ smoke test |
| `data/craigslist.py` | `external/craigslistbargain_adapter.py` | reduced adapter, no downloader/raw data | representative parser cases |
| `data/deal_or_no_deal.py` | `external/deal_or_no_deal_adapter.py` | reduced legal-allocation core | enumeration count and conservation tests |

Quantitative CSVs were transcribed from canonical valid artifacts: the complete three-solution report; the grid-fixed opponent-population analysis; context-format discovery and replication analyses; the history diagnostic; and the OpenSpiel Phase-1A audit. Release-time direct comparisons against the source checkout established the parity results above. Public tests and frozen standalone fixtures preserve those contracts without requiring access to the private working tree.
