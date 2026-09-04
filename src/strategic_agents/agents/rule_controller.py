"""Deterministic S2A baseline controller."""

DEFAULT_PRICES = {"undercut_competitors": 7.5, "hold_price": 10.0,
                  "raise_margin": 12.0, "clear_inventory": 8.5,
                  "avoid_price_war": 11.0}


class RuleController:
    def action(self, intent: str) -> float:
        if intent not in DEFAULT_PRICES: raise ValueError("unknown intent")
        return DEFAULT_PRICES[intent]
