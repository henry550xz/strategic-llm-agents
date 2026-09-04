"""Five-player repeated pricing environment (refactored from marl-controller)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class AgentObservation:
    agent_id: str
    public_state: dict[str, Any]
    private_state: dict[str, Any]
    legal_actions: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    agent_id: str
    action_type: str
    value: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observations: dict[str, AgentObservation]
    rewards: dict[str, float]
    done: bool
    info: dict[str, Any]


class PricingMarketEnv:
    """Simultaneous price competition with decentralized observations."""

    def __init__(self, num_agents: int = 5, num_rounds: int = 10,
                 base_demand: float = 100.0, unit_cost: float = 5.0,
                 min_price: float = 5.0, max_price: float = 20.0,
                 price_sensitivity: float = 1.0, reputation_weight: float = .2,
                 seed: int | None = None) -> None:
        if num_agents < 1 or num_rounds < 1 or min_price > max_price:
            raise ValueError("invalid environment configuration")
        self.num_agents, self.num_rounds = num_agents, num_rounds
        self.base_demand, self.unit_cost = float(base_demand), float(unit_cost)
        self.min_price, self.max_price = float(min_price), float(max_price)
        self.price_sensitivity, self.reputation_weight = float(price_sensitivity), float(reputation_weight)
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict[str, AgentObservation]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_round = 0
        self.price_history = {a: [] for a in self.agent_ids}
        self.profit_history = {a: [] for a in self.agent_ids}
        self.total_rewards = {a: 0.0 for a in self.agent_ids}
        noise = self.rng.uniform(-.05, .05, self.num_agents)
        self.reputation = {a: float(1 + noise[i]) for i, a in enumerate(self.agent_ids)}
        return self._observations()

    def step(self, actions: dict[str, AgentAction]) -> StepResult:
        if self.current_round >= self.num_rounds:
            raise RuntimeError("episode is complete")
        if set(actions) != set(self.agent_ids):
            raise ValueError("exactly one action is required per agent")
        prices = {}
        for agent_id in self.agent_ids:
            action = actions[agent_id]
            if action.action_type != "set_price":
                raise ValueError("only set_price is legal")
            prices[agent_id] = float(np.clip(float(action.value["price"]), self.min_price, self.max_price))
        utilities = np.array([-self.price_sensitivity * prices[a] + self.reputation_weight * self.reputation[a]
                              for a in self.agent_ids])
        mass = np.exp(utilities - utilities.max())
        shares = mass / mass.sum()
        rewards, units = {}, {}
        for i, agent_id in enumerate(self.agent_ids):
            units[agent_id] = float(shares[i] * self.base_demand)
            rewards[agent_id] = units[agent_id] * max(prices[agent_id] - self.unit_cost, 0.0)
            self.price_history[agent_id].append(prices[agent_id])
            self.profit_history[agent_id].append(rewards[agent_id])
            self.total_rewards[agent_id] += rewards[agent_id]
        self.current_round += 1
        done = self.current_round >= self.num_rounds
        return StepResult(self._observations(), rewards, done,
                          {"prices": prices, "units_sold": units, "profits": rewards,
                           "total_rewards": dict(self.total_rewards),
                           "winner": max(self.total_rewards, key=self.total_rewards.get) if done else None})

    def _observations(self) -> dict[str, AgentObservation]:
        last_prices = {a: h[-1] if h else None for a, h in self.price_history.items()}
        last_profits = {a: h[-1] if h else None for a, h in self.profit_history.items()}
        public = {"current_round": self.current_round, "num_rounds": self.num_rounds,
                  "last_prices": last_prices, "last_profits": last_profits,
                  "config": {"base_demand": self.base_demand, "unit_cost": self.unit_cost,
                             "min_price": self.min_price, "max_price": self.max_price,
                             "price_sensitivity": self.price_sensitivity,
                             "reputation_weight": self.reputation_weight}}
        return {a: AgentObservation(a, public, {"own_previous_prices": list(self.price_history[a]),
                                               "own_previous_profits": list(self.profit_history[a]),
                                               "own_reputation": self.reputation[a]}, ["set_price"])
                for a in self.agent_ids}
