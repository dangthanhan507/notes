---
layout: default
title: SAC Notes
parent: Soft Actor Critic
mathjax: true
tags: 
  - latex
  - math
has_children: true
---

# SAC Notes

For SAC, there is a warm-up phase `self.epoch_num < self.num_warmup_steps` where the agent takes random actions. In the rl_games implementation this is actions randomly sampled from `[-1, 1]` uniformly.

It's also important to note that SAC uses a replay buffer. This replay buffer is an array of past experiences denoted in $$(o_t,a_t,r_t,o_{t+1}, d_t)$$.
- $$o_t$$: observation at time t
- $$a_t$$: action taken at time t
- $$r_t$$: reward received after taking action $$a_t$$ at time t
- $$o_{t+1}$$: observation at time t+1
- $$d_t$$: done flag indicating if the episode ended after time t

This buffer is stored until it reaches capacity, at which point old experiences are discarded to make space for new ones (FIFO-style like a queue).

We sample mini-batches of experiences from the replay buffer (uniformly) and use this data to update critic networks and policy network.

We then update the critic networks which consist of two Q-functions $$Q_1$$ and $$Q_2$$. The loss function for the critic networks is defined as follows:

$$
\begin{align*}
\mathcal{L}_{\text{critic}} &= \mathbb{E}_{(o_t,a_t,r_t,o_{t+1}, d_t) \sim \mathcal{D}} \left[ \mathcal{L}_1 + \mathcal{L_2} \right] \\
\mathcal{L}_1 &= (Q_{\text{target}} - Q_1(o_t, a_t))^2 \\
\mathcal{L}_2 &= (Q_{\text{target}} - Q_2(o_t, a_t))^2 \\
Q_{\text{target}} &= r_t + \gamma (1 - d_t) V_{\text{target}} \\ 
V_{\text{target}} &= \min \{ Q_{\text{target,1}}, Q_{\text{target,2}} \} - \alpha \log \pi_{\text{target}}(\hat{a}_{t+1}|o_{t+1}) \\
\hat{a}_{t+1} &\sim \pi_{\text{target}}(\cdot|o_{t+1})
\end{align*}
$$

where there exists target networks for the critics.


Next, we update the policy network using the following:

$$
\begin{align*}

\end{align*}
$$


We still have target networks to update, so those are updated using soft updates:

$$
\begin{align*}
\theta_{\text{target}} \leftarrow \tau \theta + (1 - \tau) \theta_{\text{target}}
\end{align*}
$$