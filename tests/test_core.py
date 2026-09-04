import math
from strategic_agents.envs.pricing import AgentAction, PricingMarketEnv
from strategic_agents.evaluation.collapse import action_diagnostics
from strategic_agents.evaluation.history import intervene
from strategic_agents.evaluation.opponent_shift import cvar, summarize_opponents
from strategic_agents.evaluation.regret import exact_regret, price_grid
from strategic_agents.evaluation.replay import first_unsupported_step


def test_pricing_simultaneous_rewards_sum_to_market_profit():
    env = PricingMarketEnv(seed=3, num_rounds=1); env.reset(3)
    result = env.step({a: AgentAction(a, "set_price", {"price": 10}) for a in env.agent_ids})
    assert result.done and math.isclose(sum(result.info["units_sold"].values()), 100)
    assert math.isclose(sum(result.rewards.values()), 500)


def test_regret_grid_and_optimum():
    result = exact_regret(10, 50, .1)
    assert len(price_grid()) == 31 and result["regret"] >= 0 and result["best_price"] in price_grid()


def test_collapse_metrics():
    value = action_diagnostics([7.5]*989 + [8.0]*11)
    assert value["dominant_fraction"] == .989 and value["entropy"] > 0


def test_opponent_summary_and_cvar():
    rows = [{"opponent": "a", "regret": 1}, {"opponent": "a", "regret": 3}, {"opponent": "b", "regret": 5}]
    assert summarize_opponents(rows)["worst_opponent_regret"] == 5
    assert cvar([1, 2, 3, 4, 5], .2) == 5


def test_history_interventions():
    history = [{"step": i} for i in range(4)]
    assert intervene(history, "remove_older") == [{"step": 3}]
    shuffled = intervene(history, "shuffle_older", seed=4)
    assert shuffled[-1] == {"step": 3} and sorted(x["step"] for x in shuffled[:-1]) == [0, 1, 2]


def test_replay_support():
    assert first_unsupported_step(["a1", "a2"], ["a1", "b2"]) == 1
    assert first_unsupported_step(["a1"], ["a1"]) is None
