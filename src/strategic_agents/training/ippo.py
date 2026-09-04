"""Core parameter-sharing IPPO update functions.

Rollouts contain one trajectory per agent. Five agents act simultaneously from
decentralized observations, independently sample from the same actor, receive
their own environment rewards, and contribute transitions to this shared PPO
update.
"""
from __future__ import annotations
import numpy as np

from strategic_agents.agents.ippo_controller import PRICE_GRID, pricing_features
from strategic_agents.envs.pricing import AgentAction, PricingMarketEnv


def collect_shared_rollout(model, intent_provider, *, seed: int, num_rounds: int = 10) -> dict[str, list[dict]]:
    """Collect five simultaneous decentralized trajectories.

    ``intent_provider(agent_id, step, observation)`` supplies the semantic
    planner intent. One batched forward pass uses the shared network, while
    ``Categorical.sample`` draws each agent action independently. Each stored
    transition carries that agent's own reward and value estimate.
    """
    import torch

    env = PricingMarketEnv(num_agents=5, num_rounds=num_rounds, seed=seed)
    observations = env.reset(seed)
    trajectories = {agent_id: [] for agent_id in env.agent_ids}
    model.eval()
    for step in range(num_rounds):
        intents = {agent_id: intent_provider(agent_id, step, observations[agent_id])
                   for agent_id in env.agent_ids}
        feature_array = np.stack([pricing_features(observations[agent_id], intents[agent_id])
                                  for agent_id in env.agent_ids])
        with torch.no_grad():
            logits, values = model(torch.as_tensor(feature_array, dtype=torch.float32))
            distribution = torch.distributions.Categorical(logits=logits)
            action_indexes = distribution.sample()
            log_probabilities = distribution.log_prob(action_indexes)
        actions = {
            agent_id: AgentAction(agent_id, "set_price", {"price": PRICE_GRID[int(action_indexes[index])]})
            for index, agent_id in enumerate(env.agent_ids)
        }
        result = env.step(actions)
        for index, agent_id in enumerate(env.agent_ids):
            trajectories[agent_id].append({
                "observation": feature_array[index],
                "intent": intents[agent_id],
                "action": int(action_indexes[index]),
                "old_log_probability": float(log_probabilities[index]),
                "value": float(values[index]),
                "reward": float(result.rewards[agent_id]),
                "done": result.done,
            })
        observations = result.observations
    return trajectories


def compute_gae(rewards, values, dones, gamma: float = .99, lam: float = .95):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    next_advantage = next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t]
        next_advantage = delta + gamma * lam * mask * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def ppo_loss(model, observations, actions, old_log_probs, returns, advantages,
             clip_ratio: float = .2, value_coefficient: float = .5,
             entropy_coefficient: float = .01):
    import torch
    logits, values = model(observations)
    distribution = torch.distributions.Categorical(logits=logits)
    log_probs = distribution.log_prob(actions)
    ratios = torch.exp(log_probs-old_log_probs)
    normalized = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    surrogate = torch.minimum(ratios*normalized,
                              torch.clamp(ratios, 1-clip_ratio, 1+clip_ratio)*normalized)
    policy = -surrogate.mean()
    value = torch.nn.functional.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    return policy + value_coefficient*value - entropy_coefficient*entropy, {
        "policy_loss": policy.detach(), "value_loss": value.detach(), "entropy": entropy.detach()}


def ppo_update(model, optimizer, observations, actions, old_log_probs, returns, advantages,
               *, clip_ratio: float = .2, value_coefficient: float = .5,
               entropy_coefficient: float = .01, max_grad_norm: float = .5):
    """Apply one shared actor/critic PPO gradient update."""
    import torch

    loss, metrics = ppo_loss(model, observations, actions, old_log_probs, returns, advantages,
                             clip_ratio, value_coefficient, entropy_coefficient)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite PPO loss")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {"loss": float(loss.detach()), **{name: float(value) for name, value in metrics.items()}}
