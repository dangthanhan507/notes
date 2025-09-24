---
layout: default
title: 1. Simple Decisions
parent: Sensorimotor Learning
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
# math: katex
---

# 1. Simple Decision Making

## 1.1 Multi-armed Bandit (MAB)

This is a simple reward-based learning problem where you have to choose one out of K possible actions (discrete).

The MAB setup assumes that rewards are independent of the state and history of actions (not like how I'm used to in control). Reward depends on current action:

$$
\begin{equation}
r(a_t) = r(a_{1:t}, s_{1:t})
\end{equation}
$$

The goal is to construct a policy that maximizes reward over horizon of T time steps:

$$
\begin{equation}
\sum_{t=1}^T r_t(a_t)
\end{equation}
$$

## 1.1a Webpage example

Say we want to build a recommendation system for landing webpages. Our goal is to build a recommendation system that displays the landing page that maximizes profit. <b> NOTE: </b> We have no priors on which landing page maximizes profit.

Assume the following:
- $$T$$ - <b> Horizon </b>
- $$t$$  - <b> time step </b> or i-th user visiting a page
- $$E$$ - <b> environment </b> (website)
- $$A$$ - <b> agent </b> (recommendation system)
- $$\mathcal{A} := \{1,...,K\}$$ - <b> action space </b> where $$K$$ is the number of pages and $$k \in [1,K]$$ correspond to the action of recommending i-th page.
- $$a_t$$ - <b> action </b> at round t where $$a_t \in \mathcal{A}, \forall t \in [1, T]$$
- $$r_t(a_t) \in \mathbb{R}$$ - <b> reward </b> (random variable) which is profit earned by displaying page.
- $$T_k(t)$$ - <b> action count </b>. number of times that action $$k$$ has been taken within $$t$$ rounds.
- $$\mu_k$$ - <b> True mean reward </b> of taking action $$k$$. This optimal mean reward is defined as $$\mu^* = \max_k \mu_k$$.
- $$\hat \mu_k(t)$$ - <b> empirical mean reward </b> of taking action $$k$$, estimated by our algorithm over $$t$$ rounds.

$$
\begin{equation}
\hat \mu_k(t) := \frac{1}{T_k(t)} \sum_{t=1}^T \mathbb{I}\{a_t = k\}r(a_t)
\end{equation}
$$

where $$\mathbb{I}$$ is the indicator function. This means that when $$k$$ is chosen, the empirical mean reward is the average of the profit of the pages displayed from $$1$$ to $$T$$. 

We formulate maximizing profit as:

$$
\begin{equation}
\max_{a_1,...,a_T} \mathbb{E} [\sum_{t=1}^T r(a_t)]
\end{equation}
$$

Exploration and exploitation is the balance between trying new actions (exploration) and using the best action (exploitation).

<b> Definition 1.1 </b> (Exploration). We define any action selection $$a$$ such that:

$$
\require{cancel}
\begin{align}
\hat \mu_a(t) &\cancel{=} \max_{k} \hat \mu_k(t) \\
\hat \mu_a(t) &\cancel{=} \max_{k} \frac{1}{T_k(t)} \sum_{t=1}^T \mathbb{I}\{a_t = k \}r(a_t)
\end{align}
$$

as exploration. We are taking actions that are not the actions that maximizes the empirical mean reward. <b> NOTE: </b> this is not the same as the true mean reward. 

<b> Definition 1.2 </b> (Exploitation). We define any action selection $$a$$ such that:

$$
\begin{equation}
\hat \mu_a(t) = \max_{k} \hat \mu_k(t)
\end{equation}
$$

as exploitation. We are taking actions that maximize the empirical mean reward. 


Our goal is to try to align $$\hat \mu_a$$ and $$\mu_a$$, $$\forall a \in \mathcal{A}$$. To do this, we have to get samples of $$r(a)$$ to improve accuracy of empirical mean reward (exploration). 

Performance of an algorithm on balancing exploration and exploitation is measured by regret. In plain language, regret means "loss incurred by past actions". It characterizes gap between optimal expected reward and expected reward obtained by our algorithm. Regret over $$T$$ time steps is formally denoted as:

$$
\begin{equation}
\textbf{Regret}(T) = T\mu^*  - \mathbb{E}[\sum_{t=1}^T r(a_t)]
\end{equation}
$$

where the expectation is taken over multiple episodes of T rounds.

We can design MAB to minimize regret:

$$
\begin{equation}
\min_{a_1,...,a_T} T\mu^* - \mathbb{E}[\sum_{t=1}^T r(a_t)]
\end{equation}
$$

Minimizing regret is equivalent to maximizing rewards. Why use regret then? It actually has better theoretical properties to analyze trade-offs between exploration and exploitation. 

## 1.1.1 Explore-then-commit (ETC)

Let rewards be deterministic. Optimal strategy is try every action space once and then choose action with highest reward for all time steps in the future. 

In stochastic settings, we try each action multiple times to get a good estimate of the reward.

Assume each action is attempted $$m$$ times. Afterwards, the agent chooses to execute the optimal action with highest empirical mean reward. This is ETC. There is a distinct explore phase and exploit phase. We spend $$mK$$ time steps on exploration and $$T - mK$$ time steps on exploitation.

```python
def ETC(t,m):
    if t <= mK:
        at = (t % K) + 1
    else:
        at = np.argmax( [muhat(mK, a_k) for a_k in action_space] )
    return at
```

This is the ETC policy. Note that ```muhat``` is the empirical mean reward that changes after each call of ETC function during the exploration phase.

<b> How good is it? </b> The regret of ETC is bounded by:

$$
\begin{equation}
\textbf{Regret}(T) \leq m \sum_{k=1}^K \Delta_k + (T - mK) \sum_{k=1}^K \Delta_k \text{exp}(-\frac{m\Delta_k^2}{4})
\end{equation}
$$

where $$\Delta_k = \mu^* - \mu_k$$ is the gap between the optimal action and the action $$k$$.

This regret is split into the exploration and exploitation regret. The first term is the regret incurred during exploration and the second term is the regret incurred during exploitation.

So now we can see analyze the regret of the algorithm using big-O notation. Typically $$m := T^{\frac{2}{3}}$$, thus:

$$
\begin{equation}
\textbf{Regret}(T) \leq \text{O}(KT^{\frac{2}{3}})
\end{equation}
$$

This is better than worst regret $$\text{O}(T)$$ (linear). Worst regret is obtained if agent chooses suboptimal actions or randomly selects actions through T rounds.

## 1.1.2 Upper Confidence Bound (UCB)

The ETC exploration is "unguided". We select actions uniformly at random. If we are already certain about expected reward of an action, we should not explore it. This means we have to model uncertainty for each action to take the most promising action for exploration.

<b> Upper Confidence Bound (UCB) </b>  constructs uncertainty of actions using confidence bounds of mean reward. It employs "optimism in the face of uncertainty" principle. This principle states that we should overestimate the expected reward of actions that we are uncertain about.

We want to construct $$\hat \mu_k'$$ such that $$\hat \mu_k' \geq \mu_k$$. Finding such $$\hat \mu_k'$$ can be intractable since we do not know $$\mu_k$$.

If we assume $$r(a_t)$$ is drawn from a 1-subgaussian distribution, we can construct $$\hat \mu_k$$ probabilistically.

Remember that $$\mu_k$$ is the mean reward. Thus the following inequality holds:

$$
\begin{equation}
P(\mu_k \geq \hat \mu_k + \epsilon) \leq \text{exp}(-\frac{T_k(t)\epsilon^2}{2}), \epsilon \geq 0
\end{equation}
$$

Equivalently, solving for $$\epsilon$$ when equality holds, we transform the inequality to:

$$
\begin{equation}
P(\hat \mu_k \geq \hat \mu_k + \sqrt{\frac{2 \log \frac{1}{\delta}}{T_k(t)}}) \leq \delta, \delta \in (0,1)
\end{equation}
$$

As a result, $$\hat \mu_k$$ can be constructed by:

$$
\begin{equation}
\hat \mu_k' = \hat \mu_k + \sqrt{\frac{2\log \frac{1}{\delta}}{T_k(t)}}
\end{equation}
$$

This $$\hat \mu_k'$$ is the upper confidence bound of the empirical mean reward since $$\sqrt{\frac{2\log \frac{1}{\delta}}{T_k(t)}}$$ is a non-negative term. This means that the upper confidence bound holds with at least $$1 - \delta$$ probability.

As $$\delta$$ increases, the term $$\sqrt{\frac{2\log \frac{1}{\delta}}{T_k(t)}}$$ decreases. This means that the upper bound gap decreases as the likelihood of it holding increases. We refer to $$delta$$ as the <b> confidence level </b>.

```python
def UCB(t):
    at = np.argmax( [muhat(t, a_k) + sqrt(2*log(1/delta)/T_k(t)) for a_k in action_space] )
    return at
```
We need to make sure that setting $$\delta$$ ensures that the second term $$ \sqrt{\frac{2\log \frac{1}{\delta}}{T_k(t)}}$$ disappears as $$t$$ increases. This is the regret analysis of UCB when $$\delta = \frac{1}{t^2}$$.:

$$
\begin{equation}
\textbf{Regret}(T) \leq 8(\sum_{k=2}^K \frac{\log T}{\Delta_k}) + (1 + \frac{\pi^3}{3})(\sum_{k=1}^K \Delta_k)
\end{equation}
$$

where without loss of generality, we let $$\mu_1 = \mu^*$$ for simplicity. The big-O notation of this is:

$$
\begin{equation}
\textbf{Regret}(T) \leq \text{O}(\sqrt{KT\log T})
\end{equation}
$$

which is much tighter than ETC. 

## 1.2 Contextual Bandit (CB)

Before, we considered a special case where the reward was independent of the state. In CB, we assume that reward is state (or context) dependent. 

Now the reward is:

$$
\begin{equation}
r(a_t, s_t) = r(a_{1:t}, s_{1:t})
\end{equation}
$$

We consider the next following concepts related to CB which are augmented from MAB:
- <b> Context </b> $$x_t$$: this is the state where $$x_t \in \mathcal{X}$$ where $$\mathcal{X}$$ is the context/state space.
- <b> Contextual Reward </b> $$r(a_t, x_t) \in \mathbb{R}$$: this is the reward of taking action $$a_t$$ in context $$x_t$$.
- <b> Contextual expected regret </b> $$\textbf{Regret}(T)$$ expressed as:

$$
\begin{equation}
\textbf{Regret}(T) := \mathbb{E}[\sum_{t=1}^T \max_{a \in \mathcal{A}} r(x_t, a) - r(x_t, a_t)]
\end{equation}
$$

now how do we choose actions in CB? We can use the same principles as MAB.

## 1.2.1 LinUCB

LinUCB is a linear contextual bandit algorithm. It assumes that the reward is linear in the context.

First-pass we can try to make UCB context-dependent:

$$
\begin{equation}
\hat \mu_{x,k}(t) := \frac{1}{T_{x,k}(t)} \sum_{t=1}^T \mathbb{I}\{a_t=k, x_t=x\}r(x_t,a_t)
\end{equation}
$$

How do we obtain $$T_{x,k}(t)$$ without knowing $$\mathcal{X}$$? If we cannot enumerate through $$\mathcal{X}$$, there will be infinite elements for $$T_{x,k}(t)$$. We need to get around this issue. This is the motivation for LinUCB.

The key idea of LinUCB is to estimate context-dependent empirical mean and and confidence interval by linear regression. Let context $$x \in \mathbb{R}^d$$. We use LinUCB implementation with disjoint linear models. Let $$\hat \theta_k \in \mathbb{R}^d$$ be the parameter vector for linear regression. 

$$
\begin{equation}
\hat \mu_{\hat \theta_k}(x) := \hat \theta_k^T x
\end{equation}
$$ 

which is a scalar term.

<b> Empirical mean </b> Rewards are assumed to be drawn from a 1-subgaussian distribution. We use least-squares to get the parameter vector:

$$
\begin{equation}
\hat \theta_k^* = \arg \min_{\hat \theta_k} \mathbb{E} [(\hat \mu_{\hat \theta_k}(x) - r(x, k))^2]
\end{equation}
$$

this approximates the mean of the gaussian distribution. This can be solved in closed form using the following equation:

$$
\begin{equation}
\hat \theta_k^* = (\mathbf{D}_k^T\mathbf{D}_k + \lambda \mathbf{I})^{-1} \mathbf{D}^T_k \mathbf{c}_k
\end{equation}
$$

This is the closed form solution for linear regression but there is an added term $$\lambda \mathbf{I}$$. This is a regularization term that prevents overfitting (ridge regression).

<b> Confidence bound </b> We cannot directly construct a confidence bound because we cannot calculate $$T_{x,k}(t)$$. A workaround is needed to construct this bound:

$$
\begin{equation}
P(\mu_{x,k} \geq \hat \mu_{\hat \theta_k}(\mathbf{x}_{t,k}) + \alpha \sqrt{\mathbf{x}_{t,k}^T(\mathbf{D}_k^T\mathbf{D}_k + \lambda \mathbf{I})^{-1}\mathbf{x}_{t,k}}) \leq \delta, \delta \in (0,1)
\end{equation}
$$

where $$\alpha := 1 + \sqrt{\frac{\log\frac{2}{\delta}}{2}}$$. 

```python
def LinUCB(t, lambda, A, b):
    x_tk = receive_contexts()
    d = len(x_tk)
    theta = []
    for k in range(len(action_space)):
        ak = action_space[k]
        if ak not in taken_ks: 
            A_k = lambda * np.eye(d)
            b_k = np.zeros(d)
            A[k] = A_k
            b[k] = b_k
            taken_ks.add(ak)
        theta_k = np.linalg.inv(A[k]) @ b[k]
        theta.append(theta_k)
    alpha = 1 + np.sqrt(2 * np.log(2 / delta) / 2)
    at = np.argmax( [theta[k] @ x_tk + alpha * sqrt(x_tk.T @ np.linalg.inv(A[k]) @ x_tk) for k in action_space] )

    # modify A, b for next calls of LinUCB
    A[k] = A[k] + np.outer(x_tk, x_tk)
    b[k] = b[k] + r(x_tk, at) * x_tk
    return at
```

