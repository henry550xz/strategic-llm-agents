"""Paired, group-stratified episode bootstrap."""
from collections import defaultdict
import numpy as np


def paired_stratified_ci(rows: list[dict], *, group="opponent", pair="episode", left="left",
                         right="right", iterations=5000, seed=0) -> tuple[float, float]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[group]].append(float(row[left]) - float(row[right]))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sampled = []
        for values in buckets.values():
            sampled.extend(rng.choice(values, size=len(values), replace=True))
        estimates.append(float(np.mean(sampled)))
    return tuple(float(x) for x in np.quantile(estimates, [.025, .975]))
