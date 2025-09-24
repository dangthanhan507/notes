---
layout: default
title: Math Notes
parent: Graph Insertion
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---

# Quick RL Math Notes

## Advantage Actor Critic


$$
\begin{equation}
R^\gamma(\tau_{t:T}) = Q^\pi(s_t,a_t) = \mathbb{E} [ \sum_{i=t}^T \gamma^{i-t} r(s_i, a_i) \mid s_t, a_t  ]
\end{equation}
$$

- $$R$$ is returns function ($$\gamma$$ is discount factor but not exponent of $$R$$)
- $$Q$$ is the action-value function
- $$r$$ is the reward function.

When calculating policy gradients, we get them from solving the following optimization problem:

$$
\begin{equation}
\max_{d} J(\theta + d) - J(\theta) \\
\text{s.t.} \quad \mathbb{E}_s \left[ D_{KL}(\pi_\theta(a \mid s) || \pi_{\theta + d}(a \mid s)) \right] \leq \epsilon
\end{equation}
$$

- $$J(\theta)$$ = $$\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi(a \mid s_t^t) A(s_t,a_t)$$
- $$A(s_t,a_t)$$ = $$Q(s_t,a_t) - V(s_t)$$
- $$\mathbf{F}(\theta)$$ = $$\mathbb{E}_{\pi_\theta (a \mid s)} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^T \right]$$ this is the Fisher information matrix (second-order approx of KL-divergence).

Using our knowledge of Negative Log-Likelihood, without the advantage function, we are trying to use our neural network to match the action distribution from our rollouts (maximizing log-likelihood of actions from rollouts). And our cost function is the gradient of the log-likelihood (doing gradient ascent). We are just maximizing the cost boost without stepping too far away from the action distribution (KL-divergence constraint).

Since our actions are delta-poses, we can almost think of the KL-divergence constraint on a rollout trajectory to create a funnel of admissible trajectories (actions). 

Using the insertion reward (keypoints), it's easy to see how when we sample from the network which actions should be preferred (actions taken closer to the goal). 

If we use this other reward (packed object arrangement), there is no direct correlation between an action taken and the reward improving. 
- This motivates the case for curriculum learning, we generate trajectories close to the goal so that we can find pushing actions from the funnel of admissible actions to push occluding objects out of the way. 