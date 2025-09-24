---
layout: default
title: 2. Sequential Decisions
parent: Sensorimotor Learning
nav_order: 2
mathjax: true
tags: 
  - latex
  - math
# math: katex
---

# 2. Sequential Decision Making

Sequential decision making is when an agent must make a horizon or sequence of decisions over time instead of a single decision. 

## 2.1 Issues of CB

The thing with CB is that it does not consider the context transitions affected by actions. In control theory, we refer to this as the dynamics of the system. Previously, we just separated the state and action spaces. 


## 2.2 Sequential Decision Process

Note that this is broader than Markov Decision Processes (MDPs) because it does not require the Markov property.

- **Timestep**: $$t$$
- **State**: $$s_t \in \mathcal{S}$$
- **Action**: $$a_t \in \mathcal{A}$$
- **Reward**: $$r(s_{1:t}, a_{1:t}) \in \mathbb{R}$$
- **Transition**: $$\mathcal{T}(s_{t+1}\mid s_{1:t,a_{1:t}})$$ non-markov
- **Policy**: $$\pi$$ where $$a_t = \pi(s_{1:t}, a_{1:t-1})$$ Sometimes we treat $$\pi : (\mathcal{S} \times \mathcal{A})^t \mapsto [0,1]^{\mid \mathcal{A} \mid }$$

The objective is to maximize the expected reward over a horizon $$T$$:

$$
\begin{equation}
\max_{\pi} \mathbb{E}_{s_{1} \sim P(\mathcal{S}), a_t \sim \pi(s_{1:t}), s_{t+1} \sim \mathcal{T}(s' \mid s_{1:t}, a_{1:t})} [\sum_{t=1}^T r(s_{1:t}, a_{1:t})]
\end{equation}
$$

where $$P(\mathcal{S})$$ is the initial state distribution. We refer to sum of rewards as the **return**.

## 2.3 Finite and Infinite Horizon

We have been thinking about the finite horizon case. For infinite-horizon, it is much trickier. Especially when $$T \to \infty$$ our rewards can diverge. We need to make sure that the rewards are bounded.

We can do this by using a discount factor and restate the objective as:

$$
\begin{equation}
\max_{\pi} \mathbb{E} [\sum_{t=1}^\infty \gamma^{t-1} r(s_t, a_t)]
\end{equation}
$$

where $$\gamma$$ denotes the discount factor. We want it to be between 0 and 1. This is a common trick in reinforcement learning.

This is definitely a heuristic on how far we should care about the future. If $$\gamma \to 0^{+}$$, we only care about the immediate reward. If $$\gamma \to 1^{-}$$, we care about all future rewards equally.

## 2.4 Trajectory and Episode

<b> Rolling out </b> means to execute a policy in the environment. What you get from rolling out is a <b> rollout </b> or  <b>trajectory </b> (rollout for T steps). Process of sampling a single trajectory is an <b> episode </b>. 

This is a trajectory (episode):

$$
\begin{equation}
\tau_{1:T}^i := (s_1^i, a_1^i, r_1^i, s_2^i, a_2^i, r_2^i, \ldots, s_T^i, a_T^i, r_T^i)
\end{equation}
$$

where $$i$$ is the index of the episode.

