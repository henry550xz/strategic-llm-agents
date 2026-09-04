#!/usr/bin/env python3
"""Run the real Pricing environment and exact regret evaluator on CPU."""
from strategic_agents.envs.pricing import AgentAction, PricingMarketEnv
from strategic_agents.evaluation.collapse import action_diagnostics
from strategic_agents.evaluation.regret import exact_regret

env = PricingMarketEnv(seed=7, num_rounds=4)
observations = env.reset(7)
chosen, regrets, focal_rewards = [], [], []
policies = {"agent_0": [9.5, 9.0, 8.5, 8.0], "agent_1": [10.0]*4,
            "agent_2": [11.0]*4, "agent_3": [8.0, 8.5, 9.0, 9.5], "agent_4": [12.0]*4}
for step in range(4):
    actions = {a: AgentAction(a, "set_price", {"price": prices[step]}) for a, prices in policies.items()}
    result = env.step(actions)
    p, reward = policies["agent_0"][step], result.rewards["agent_0"]
    share = result.info["units_sold"]["agent_0"] / env.base_demand
    regrets.append(exact_regret(p, reward, share)["regret"])
    focal_rewards.append(reward)
    chosen.append(p)
print({"demo": "deterministic CPU execution", "focal_return": round(sum(focal_rewards), 6),
       "mean_one_step_regret": round(sum(regrets)/len(regrets), 6), "behavior": action_diagnostics(chosen)})
