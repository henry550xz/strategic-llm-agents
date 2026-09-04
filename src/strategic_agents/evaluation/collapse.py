"""Behavioral diagnostics that complement scalar reward and regret."""
from collections import Counter
from math import log


def action_diagnostics(actions: list[float | int | str]) -> dict[str, float | int | str | None]:
    if not actions:
        return {"count": 0, "unique_actions": 0, "dominant_action": None,
                "dominant_fraction": 0.0, "entropy": 0.0}
    counts = Counter(actions)
    n = len(actions)
    dominant, frequency = counts.most_common(1)[0]
    entropy = -sum((c/n) * log(c/n) for c in counts.values())
    return {"count": n, "unique_actions": len(counts), "dominant_action": dominant,
            "dominant_fraction": frequency/n, "entropy": entropy}
