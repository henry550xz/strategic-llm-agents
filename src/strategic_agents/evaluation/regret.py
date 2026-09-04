"""Exact one-step unilateral regret in the Pricing environment."""
from math import exp


def price_grid(low: float = 5.0, high: float = 20.0, step: float = .5) -> tuple[float, ...]:
    if step <= 0 or high < low:
        raise ValueError("invalid grid")
    return tuple(round(low + i * step, 10) for i in range(int(round((high-low)/step)) + 1))


def counterfactual_reward(chosen_price: float, chosen_share: float, candidate_price: float,
                          base_demand: float = 100.0, unit_cost: float = 5.0,
                          price_sensitivity: float = 1.0) -> float:
    """Hold opponents fixed and change only the focal price."""
    multiplier = exp(-price_sensitivity * (candidate_price - chosen_price))
    share = multiplier * chosen_share / (1 - chosen_share + multiplier * chosen_share)
    return base_demand * share * max(candidate_price - unit_cost, 0.0)


def exact_regret(chosen_price: float, chosen_reward: float, chosen_share: float,
                 grid: tuple[float, ...] | None = None, **market: float) -> dict[str, float]:
    candidates = grid or price_grid()
    rewards = [(p, counterfactual_reward(chosen_price, chosen_share, p, **market)) for p in candidates]
    best_price, best_reward = max(rewards, key=lambda x: x[1])
    return {"chosen_reward": chosen_reward, "best_price": best_price,
            "best_reward": best_reward, "regret": max(0.0, best_reward-chosen_reward)}
