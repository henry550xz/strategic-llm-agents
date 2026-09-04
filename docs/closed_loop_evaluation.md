# Closed-loop evaluation from logged human trajectories

Human dialogue datasets provide factual trajectories, not counterfactual simulators:

```text
logged human trajectory: h1 -> a1 -> h2 -> a2 -> h3 -> a3 -> h4
teacher forced:          h1 -> a1 -> h2 -> a2 -> h3 -> a3
closed loop:             h1 -> a1 -> h2 -> b2 -> ?
```

Teacher-forced next-action evaluation conditions on states that actually occurred. If a model chooses (b_2\ne a_2), the dataset generally contains no real human counterpart response conditioned on that deviation. Recursively generating both sides creates a new state distribution; model errors can compound through the other agents' reactions.

Consequently, next-action accuracy measures on-trajectory response modeling. It does not, by itself, validate a recursively generated synthetic population, partner adaptation, or long-horizon strategic competence.

This distinction arose while integrating CraigslistBargain and Deal-or-No-Deal. Both are valuable logged interaction datasets, and the adapters expose structured actions and legal allocations. Neither dataset supplies the counterfactual transition mechanism required to treat S2B MARL or synthetic negotiation rollouts as ground-truth online interaction.

The executable [`demo_replay_vs_closed_loop.py`](../scripts/demo_replay_vs_closed_loop.py) marks the first generated deviation as leaving logged support. It deliberately does not fabricate a human-response model. Learning or validating such a response model—and quantifying uncertainty off logged support—remains a research question.
