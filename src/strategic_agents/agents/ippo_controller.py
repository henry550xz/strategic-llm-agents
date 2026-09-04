"""Intent-conditioned parameter-sharing actor-critic used by S2B."""
from __future__ import annotations
from typing import Any
import numpy as np

PRICE_GRID = tuple(5.0 + .5 * i for i in range(31))
INTENTS = ("undercut_competitors", "hold_price", "raise_margin", "clear_inventory", "avoid_price_war")


def pricing_features(observation: Any, intent: str) -> np.ndarray:
    """Eleven decentralized features plus a five-way semantic intent."""
    if intent not in INTENTS:
        raise ValueError("unknown intent")
    public, private = observation.public_state, observation.private_state
    cfg = public["config"]
    low, high, cost = float(cfg["min_price"]), float(cfg["max_price"]), float(cfg["unit_cost"])
    span = max(high-low, 1e-9)
    own_p = [float(x) for x in private.get("own_previous_prices", [])]
    own_r = [float(x) for x in private.get("own_previous_profits", [])]
    midpoint = (low+high)/2
    competitors = [float(v) for k, v in public["last_prices"].items()
                   if k != observation.agent_id and v is not None] or [midpoint]
    competitor_rewards = [float(v) for k, v in public["last_profits"].items()
                          if k != observation.agent_id and v is not None] or [0.0]
    scale = max(float(cfg["base_demand"]) * max(high-cost, 1), 1)
    base = [float(public["current_round"])/max(float(public["num_rounds"]), 1),
            ((own_p[-1] if own_p else midpoint)-low)/span,
            ((own_p[-1]-own_p[-2]) if len(own_p) > 1 else 0)/span,
            (own_r[-1] if own_r else 0)/scale, float(private["own_reputation"]),
            (min(competitors)-low)/span, (float(np.mean(competitors))-low)/span,
            (max(competitors)-low)/span, float(np.mean(competitor_rewards))/scale,
            max(competitor_rewards)/scale, float(bool(own_p))]
    one_hot = [float(name == intent) for name in INTENTS]
    return np.asarray(base + one_hot, dtype=np.float32)


class IntentActorCritic:
    """Shared 2x64 tanh actor/critic; lazily imports Torch."""
    def __new__(cls, input_dim: int = 16, hidden_dim: int = 64, action_count: int = 31):
        import torch.nn as nn

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
                self.actor = nn.Linear(hidden_dim, action_count)
                self.critic = nn.Linear(hidden_dim, 1)

            def forward(self, observations):
                hidden = self.trunk(observations)
                return self.actor(hidden), self.critic(hidden).squeeze(-1)
        return Network()
