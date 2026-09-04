"""Opponent-stratified evaluation summaries."""
from collections import defaultdict
from statistics import mean


def cvar(values: list[float], tail_fraction: float = .2) -> float:
    """Upper-tail CVaR for losses such as regret."""
    if not values:
        raise ValueError("values cannot be empty")
    count = max(1, int(len(values) * tail_fraction + .999999))
    return mean(sorted(values, reverse=True)[:count])


def summarize_opponents(rows: list[dict], regret_key: str = "regret") -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["opponent"]].append(float(row[regret_key]))
    by_opponent = {name: mean(values) for name, values in sorted(groups.items())}
    return {"mean_regret": mean(float(r[regret_key]) for r in rows),
            "worst_opponent_regret": max(by_opponent.values()),
            "cvar20_regret": cvar([float(r[regret_key]) for r in rows]),
            "by_opponent": by_opponent}
