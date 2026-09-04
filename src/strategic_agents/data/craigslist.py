"""Minimal CraigslistBargain action adapter.

Obtain data from the original Stanford CoCoA/CraigslistBargain distribution.
This package does not download or redistribute it.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class BargainAction:
    kind: str
    price: float | None
    text: str


def parse_turn(text: str) -> BargainAction:
    lowered = text.lower()
    match = re.search(r"\$?([0-9]+(?:\.[0-9]+)?)", text)
    price = float(match.group(1)) if match else None
    if any(x in lowered for x in ("accept", "deal", "agreed")): kind = "accept"
    elif any(x in lowered for x in ("reject", "no thanks", "too low")): kind = "reject"
    elif price is not None: kind = "offer"
    else: kind = "message"
    return BargainAction(kind, price, text)
