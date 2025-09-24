---
layout: default
title: 4. Reward Function
parent: Sensorimotor Learning
nav_order: 4
mathjax: true
tags: 
  - latex
  - math
# math: katex
---

# 4. Reward Function Design

- Supervised learning we solve optimization from data:

$$
\begin{equation*}
\min_f \mathbb{E} [(y-f(x))^2] 
\end{equation*}
$$

- Policy gradient we solve optimization:

$$
\begin{equation*}
\max_\theta \mathbb{E} [\sum_t^T \nabla_\theta \log \pi_\theta (a \mid s_t) R(\tau_{t:T})]
\end{equation*}
$$

we can see that $$R(\tau_{t:T})$$ is sampled by the policy instead of a fixed dataset.

## 5.1 Challenges of Reward Function Design

<b> Sparse Reward </b>

$$
\begin{equation*}
r(s,a) = \begin{cases}
1, & \mathcal{T}(s' \mid s,a) = s_g \\
0, & \text{otherwise}
\end{cases}
\end{equation*}
$$

This specifies reaching the goal but is too sparse to optimize for policy gradient. 

<b> Dense Reward </b> To address reward sparsity, an alternative is to use distance to the goal as reward for encouraging agent to approach goal (MPC-style cost function).

$$
\begin{equation*}
r(s,a) = \begin{cases}
-\| s_g - s' \|_2 + 1, & s' = s_g, s' = \mathcal{T}(s' \mid s,a) \\
-\| s_g - s' \|_2, & \text{otherwise}
\end{cases}
\end{equation*}
$$

<b> Oracle Reward </b> However, if we're doing things like obstacle avoidance, we have to take a detour which is not reflected in this reward function. We will call this detour distance the geodesic distance and we will have to add ($$D_g$$) that into our reward function:

$$
\begin{equation*}
r(s,a) = \begin{cases}
-D_g(s', s_g) + 1, & s' = s_g, s' = \mathcal{T}(s' \mid s,a) \\
-D_g(s', s_g), & \text{otherwise}
\end{cases}
\end{equation*}
$$

Getting geodesic distance is hard, and we sometimes need to get enough samples to estimate the geodesic distance.

Here's the summary of 3 reward function design:
- **Sparse reward**: gives the correct reward but is too sparse to optimize for policy gradient.
- **Dense reward**: easy to optimize but gives wrong objective to optimize. 
- **Oracle reward**: ideal reward function which is correct and easy to optimize. 

## 5.2 Reward Function Engineering

<b> Achieve goal as soon as possible </b>

$$
\begin{align*}
r(s,a) &= \begin{cases}
1, & \mathcal{T}(s' \mid s,a) = s_g \\
-0.1 & \text{otherwise}
\end{cases} \\
r(s,a) &= - \| a \|_2
\end{align*}
$$

We punish how many state transitions occur (try to finish as soon as possible) and we also regularize the action to be small.

<b> Avoid danger </b> 

$$
\begin{equation*}
r(s,a) = \begin{cases}
1, & \mathcal{T}(s' \mid s,a) = s_g \\
-0.3 & \mathcal{T}(s' \mid s,a) = \text{Walls} \\
-0.1 & \text{otherwise}
\end{cases}
\end{equation*}
$$

we add penalty for hitting walls. this can be considered sparse reward, and we can do a much more dense reward function by adding a distance to the wall (but be careful as the goal is not to be far from the wall, but to avoid it). We can be close to the wall as long as we don't hit it (maybe it helps to get an optimal policy to be close to the wall).

## 5.3 Reward Shaping

$$
\begin{equation*}
r'(s,a) = r(s_t, a_t) + \phi(s_t) - \phi(s_{t-1})
\end{equation*}
$$

where $$\phi(s)$$ is a potential function. The shapes reward function is unbiased and gives rise to optimal policy invariant to unshaped reward function. This is because $$\phi(s') - \phi(s)$$ cancels out when summing over time.

This principle of reward shaping has not been shown to be useful in practice. 