"""Interventions for separating history sensitivity from useful inference."""
from __future__ import annotations
import random


def intervene(history: list[dict], mode: str, *, seed: int = 0, keep_recent: int = 1) -> list[dict]:
    rows = [dict(x) for x in history]
    if mode == "intact":
        return rows
    split = max(0, len(rows) - keep_recent)
    if mode == "remove_older":
        return rows[split:]
    if mode == "shuffle_older":
        older = rows[:split]
        random.Random(seed).shuffle(older)
        return older + rows[split:]
    raise ValueError(f"unknown intervention: {mode}")
