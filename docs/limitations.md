# Limitations

- The matched four-architecture Pricing comparison contains two episodes and is diagnostic only.
- Pricing admits a strong near-constant shortcut; low regret does not establish opponent modeling.
- The population-shift comparison is based on evaluation episodes for one training run per condition, not training-seed inference.
- The context-format result is cleanly replicated only for the Pricing-trained Qwen family. Two other instruction-model families collapsed to constant actions or failed competence gates.
- History interventions establish behavioral dependence, not beneficial inference.
- OpenSpiel support is evaluation infrastructure, not a completed large-scale learned-policy benchmark.
- Logged CraigslistBargain and Deal-or-No-Deal trajectories do not identify human responses on synthetic counterfactual branches.
- Training local SFT/GRPO policies requires a separate GPU environment; CPU demos verify environments and evaluators only.
