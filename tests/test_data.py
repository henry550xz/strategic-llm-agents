from strategic_agents.data.craigslist import parse_turn
from strategic_agents.data.deal_or_no_deal import legal_allocations, utility


def test_craigslist_parse():
    action = parse_turn("I can do $25")
    assert action.kind == "offer" and action.price == 25


def test_deal_allocations():
    rows = list(legal_allocations([1, 2, 1]))
    assert len(rows) == 12 and all(x+y == total for row in rows for x, y, total in zip(row["agent_0"], row["agent_1"], [1, 2, 1]))
    assert utility([1, 0, 1], [3, 2, 5]) == 8
