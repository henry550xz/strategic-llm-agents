import copy


def redundant(canonical: dict) -> dict:
    value = copy.deepcopy(canonical)
    value["redundant_equivalent_view"] = {
        "current_market": {"round": value["current_round"], "last_prices": value["last_prices"], "last_profits": value["last_profits"]},
        "trajectory_facts": copy.deepcopy(value["recent_history"]),
        "focal_recent": {"prices": value["own_previous_prices"], "profits": value["own_previous_profits"]}}
    return value
