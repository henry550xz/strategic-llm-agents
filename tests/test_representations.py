from strategic_agents.representations import compact, extract_semantics, redundant, verbose


def state():
    return {"current_round": 1, "last_prices": {"a": 9.5}, "last_profits": {"a": 20},
            "own_previous_prices": [9.5], "own_previous_profits": [20],
            "recent_history": [{"step": 0, "prices": {"a": 9.5}, "rewards": {"a": 20}, "done": False}]}


def test_all_lossless():
    source = state()
    assert extract_semantics(compact(source), "compact") == source
    assert extract_semantics(verbose(source), "verbose") == source
    assert extract_semantics(redundant(source), "redundant") == source


def test_redundant_is_longer():
    assert len(str(redundant(state()))) > len(str(verbose(state())))
