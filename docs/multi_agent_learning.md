# Multi-agent learning

## S2B: semantic planning with parameter-sharing IPPO

S2B separates a slow semantic planner from a learned controller. Every two decisions, the planner maps state and permitted history to one of five intents. Each agent then concatenates its intent one-hot vector with an 11-dimensional decentralized observation and independently samples a price from a shared categorical policy.

Five agents learn simultaneously. They share the same two-layer, 64-unit tanh trunk, actor head, and critic head, but no agent receives another agent's private observation. For agent (i),

\[
a_t^i \sim \pi_\theta(\cdot\mid o_t^i,z_t^i), \qquad
r_t^i = \text{PricingProfit}_i(a_t^1,\ldots,a_t^5).
\]

Parameter sharing means every transition updates the same parameters (\theta), while independent sampling and per-agent trajectories preserve decentralized execution. The implementation is directly inspectable in [`ippo_controller.py`](../src/strategic_agents/agents/ippo_controller.py) and [`ippo.py`](../src/strategic_agents/training/ippo.py).

## PPO and GAE

With probability ratio (r_t(\theta)=\pi_\theta(a_t\mid o_t,z_t)/\pi_{\theta_{old}}(a_t\mid o_t,z_t)), the clipped policy objective is

\[
L^{clip}(\theta)=\mathbb E_t\left[\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right].
\]

Advantages use generalized advantage estimation:

\[
\delta_t=r_t+\gamma V(o_{t+1},z_{t+1})-V(o_t,z_t),\qquad
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
\]

The optimized loss combines negative clipped return, squared value error, entropy regularization, and an auxiliary intent-alignment cross-entropy term. The original full runs used 25,000 joint environment steps per seed (125,000 agent transitions), PPO/GAE, environment profit scaled by 0.01, and a 0.5 intent auxiliary coefficient. Rule-policy imitation initialized the network; rollout and deployment actions came from the learned controller.

## What is established

The implementation passed finite-loss, parameter-change, action-legality, save/load, decentralized-observation, independent-sampling, and intent-sensitivity checks. It demonstrates a functioning learned multi-agent controller—not statistical superiority. Robustness, causal benefit over the rule controller, and transfer beyond Pricing remain open.
