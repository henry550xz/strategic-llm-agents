import numpy as np
from strategic_agents.training.ippo import compute_gae
from strategic_agents.training.grpo import canonical_price_action, parse_price_action


def test_gae_terminal_boundaries():
    advantages, returns = compute_gae([1, 2], [0.5, .25], [False, True], gamma=.9, lam=.8)
    assert np.allclose(advantages, [1.985, 1.75], atol=1e-6)
    assert np.allclose(returns, [2.485, 2.0], atol=1e-6)


def test_grpo_action_support_is_exact():
    assert parse_price_action(canonical_price_action(7.5)) == 7.5
    for invalid in (4.5, 7.1, 20.5):
        try: canonical_price_action(invalid)
        except ValueError: pass
        else: raise AssertionError("invalid optimizer action accepted")
