"""Legal-allocation helpers for Deal-or-No-Deal dialogue data."""
from itertools import product


def legal_allocations(item_counts: list[int]):
    for own in product(*(range(count + 1) for count in item_counts)):
        yield {"agent_0": list(own), "agent_1": [count-x for count, x in zip(item_counts, own)]}


def utility(allocation: list[int], private_values: list[int]) -> int:
    return sum(x*v for x, v in zip(allocation, private_values))
