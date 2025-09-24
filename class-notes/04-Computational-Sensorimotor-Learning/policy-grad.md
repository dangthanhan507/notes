---
layout: default
title: 3. Policy Gradients
parent: Sensorimotor Learning
nav_order: 3
mathjax: true
tags: 
  - latex
  - math
# math: katex
---

# 3. Policy Gradients

Policy gradients are used to solve SDM problems. We take the gradient of the sum of rewards with respect to policy parameters. Using gradient ascent, we can find the optimal policy. 

## 3.1 Derivation and Properties of Policy Gradient

$$
\begin{equation}
\max_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}} \left[ R^{\gamma}(\tau) \right]
\end{equation}
$$

where $$R^{\gamma}(\tau)$$ denotes the discounted return of a trajectory $$\tau$$ and $$\pi_{\theta}$$ denotes policy parametrized by $$\theta$$.

Let $$p_{\theta}(\tau)$$ be probability of trajectory $$\tau$$ being sampled. We express gradient of expected return $$\nabla_{\theta} J(\theta)$$ as:

$$
\begin{align*}
\nabla_{\theta} J(\theta) &= \nabla_{\theta} \mathbb{E}_{\pi_{\theta}} \left[ R^{\gamma}(\tau) \right] \\
&= \nabla_{\theta} \int_{\tau} p_{\theta}(\tau) R^{\gamma}(\tau) d\tau \\
&= \int_{\tau} \nabla_{\theta} p_{\theta}(\tau) R^{\gamma}(\tau) d\tau \\
&= \int_{\tau} p_{\theta}(\tau) \frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta}(\tau)} R^{\gamma}(\tau) d\tau \\
&= \int_{\tau} p_{\theta}(\tau) \nabla_{\theta} \log p_{\theta}(\tau) R^{\gamma}(\tau) d\tau \\
&= \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log (p_\theta (\tau)) R^\gamma (\tau) \right]
\end{align*}
$$

using the log-derivative trick, we can move the gradient inside the expectation. Now the calculation of the gradient is reduced to calculating the gradient of the log-likelihood of the trajectory. Gradient ascent can be calculated as follows $$ \theta \gets \theta + \alpha \nabla_{\theta} J(\theta)$$.



<b> Remark 3.1 </b> (Continuous action spaces). Policy gradients make no assumption about action spaces. We can use any choice of probabilistic distribution for $$\pi$$ and sample $$a_t$$ from $$\pi$$.

<b> Remark 3.2 </b> (Model-free). We can rewrite $$\nabla_{\theta} J(\theta)$$ as:

$$
\begin{equation}
\mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \log (p_\theta (\tau)) R^\gamma (\tau) \right] = \mathbb{E}_{\pi_{\theta}} \left[ \nabla_{\theta} \sum_{t=1}^T \log (\pi_\theta (a_t \mid s_{1:t}, a_{1:t}))R^\gamma(\tau_{1:T}) \right]
\end{equation}
$$

We can see that this is independent of the state transition model $$P$$. We say that this is model-free. This means that we estimate policy gradients by sampling trajectories from an environment. 

<b> Remark 3.3 </b> (No Markov Assumption). Markov assumption is described as follows:

$$
\begin{equation}
\pi(a_t \mid s_{1:t}, a_{1:t}) = \pi(a_t \mid s_t)
\end{equation}
$$

a lot of algorithms assume that the policy is Markovian. However, this is not necessary in this case. We can make our policy gradient algorithm work with Markovian policies if we want to.

## 3.2 REINFORCE

The above policy gradient derivation is also known as REINFORCE. However, we just described the update step. These are the exact steps of the algorithm:

1. Roll out trajectory $$\tau^i$$ using policy $$\pi_{\theta}$$.
2. Compute gradient of log-likelihood of all actions $$g^i = \nabla_\theta \log \pi_\theta (\cdot \mid \cdot)$$.
3. Weigh gradient by corresponding returns $$g^i R^\gamma (\tau^i)$$.
4. Update policy parameters $$\theta \gets \theta + \alpha g^i R^\gamma (\tau^i)$$.

We can rollout multiple trajectories and average the gradients. This is called the Monte Carlo estimate of the gradient.

If we assume markov property, we can rewrite the steps as:
1. Collect $$N$$ trajectories $$\{ \tau_{1:T}^i \}_{i=1}^N$$ with policy $$\pi_{\theta}$$.
2. Compute gradient $$\nabla_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi (a \mid s^i_t) R^\gamma (\tau^i_{1:t})$$.
3. update policy parameters $$\theta \gets \theta + \alpha \nabla_\theta J(\theta)$$.

## 3.3 Credit Assignment

The idea of weighing the gradient of the policy by the return is called credit assignment. Unfortunately, credit assignment is not easy because of the high variance in sampled trajectory returns $$R^\gamma (\tau)$$. The reason why this is the case is that it is challenging to distinguish contribution of eacha ction to the observed return. This is one of the issues that come with model-free approaches. Normally in model-based approaches, the gradient of the transition model can allow us to figure out the contribution of each action to the observed return.

The issue still stands and it only gets worse as trajectory length increases.

<b> Discount </b> We can use discount factor to reduce the variance of the return. A lower discount factor anneals the variance of the return since the total magnitude of the return is reduced. However, too small of a discount can lead to suboptimal policies. 

<b> Baseline </b> We can introduce a baseline to increase probability of trajectories with above-average returns and decrease that with below-average returns (baseline is average return). We increase probabilities of trajectories with "advantage". Advantage quantifies gap between return of trajectory $$R^\gamma (\tau)$$ to the average trajectory return $$b(s_t) = \mathbb{E}_{\pi_{\theta}} \left[ R^\gamma (\tau) \right]$$. With notion of advantage, we re-write policy gradient as:

$$
\begin{equation}
\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi (a \mid s_t^i) (R^\gamma (\tau^i_{1:t}) - b(s_t^i))
\end{equation}
$$

where $$R^\gamma (\tau^i_{1:t}) - b(s_t^i)$$ is the advantage of the action $$a_t$$.

A fundamental reason that baselines reduce variance is because of the following inequality:

$$
\begin{equation}
\text{Var} \left[ \nabla_\theta \log(p_\theta(\tau))(R^\gamma(\tau) - b(\tau)) \right] \leq \text{Var} \left[ \nabla_\theta \log(p_\theta(\tau))R^\gamma(\tau) \right]
\end{equation}
$$

which only holds when $$R^\gamma(\tau)$$ and $$b(\tau)$$ are correlated. If we assume both of these are random variables, then the variance of the difference of two random variables is described as:

$$
\begin{equation}
\text{Var} \left[ X - Y \right] = \text{Var} \left[ X \right] + \text{Var} \left[ Y \right] - 2\text{Cov} \left[ X, Y \right]
\end{equation}
$$

if $$2\text{Cov} \left[ X, Y \right] > \text{Var} \left[ Y \right]$$ (i.e. X and Y correlate), then $$\text{Var} \left[ X - Y \right] < \text{Var} \left[ X \right]$$. This is because the term subtracts from the variance of $$X$$ more than the contribution of the variance of $$Y$$.

Here are some common choices of baselines:
- **Weighted average**: $$b(s_t) = \frac{\mathbb{E} \left[ \| \nabla_\theta \log \pi_\theta(a \mid s_t) \|_2^2 R(\tau_{1:T}) \right]}{ \| \mathbb{E} \left[ \nabla_\theta \log \pi_\theta (a \mid s_t) \|_2^2 \right] }$$
- **Value function**: $$ V^\pi (s_t) = \mathbb{E}_{\pi_\theta} \left[ R^\gamma (\tau_{1:T}) \right]$$

Calculating the value function is a bit tricky. We can parametrize the value function with another model (separate from policy model) $$V^\pi_\psi$$. We then minimize the expected magnitude of advantage with respect to $$\psi$$:

$$
\begin{equation}
\min_\psi \mathbb{E} \left[ \| R^\gamma(\tau_{t:T}) - V^\pi_\psi (s_t) \|_2^2 \right]
\end{equation}
$$

where $$R^\gamma(\tau_{t:T})$$ is the monte carlo estimation of expected return of a trajectory starting from $$s_t$$. 

<b> Sample size </b> Another way to reduce variance is to increase number of trajectories sampled $$N$$. However, we know that this is not always possible and can be too expensive, especially considering that we are sampling $$N$$ trajectories for each gradient update.

## 3.4 Actor-Critic
Actor-critic method uses the value function to replace MC estimates of trajectory returns $$R^\gamma(\tau)$$ and reduce variance of the policy gradient. This section goes over the <b> Advantage Actor-Critic </b> method and the <b> Generalized Advantage Estimation </b> (GAE) method which balances bias-variance trade-off between MC estimation and value predictions.

## 3.4.1 Advantage Actor-Critic (A2C)

A2C smooths estimates of traj return by averaging which reduces variance of policy gradient. The reason to smooth is because it is unlikely to obtain samples of trajectory returns from exactly the same state. As a result, we won't get sufficient samples of trajectory returns starting from the same state.

Taking average trajectory returns from similar states will reduce the variance of estimating the trajectory return starting from state $$s$$.

To smooth estimates of $$R^\gamma(\tau)$$, A2C replaces the MC estimates of trajectory returns $$R^\gamma(\tau)$$ starting from state $$s_t$$ with:

$$
\begin{equation}
R^\gamma(\tau_{t:T}) = Q^\pi(s_t,a_t) = \mathbb{E} [ \sum_{i=t}^T \gamma^{i-t} r(s_i, a_i) \mid s_t, a_t  ]
\end{equation}
$$

where $$Q$$ is the expected return of taking action $$a_t$$ in state $$s_t$$. $$\mathbb{E}[\cdot \mid s_t, a_t]$$ means $$s_t$$ and $$a_t$$ are fixed. $$Q$$ can be obtained from the value function baseline $$V^\pi$$ by constructing $$Q^\pi$$ in a recursive way:

$$
\begin{align*}
Q^\pi(s_t,a_t) &= \mathbb{E} \left[ \sum_{i=t}^T \gamma^{i-t} r(s_i,a_i) \mid s_t, a_t \right] \\
&= \mathbb{E} \left[ r(s_t, a_t) + \gamma V^\pi(s_{t+1}) \right] \\
&\approx \mathbb{E} \left[ r(s_t,a_t) + \gamma V^\pi_\psi (s_{t+1}) \right]
\end{align*}
$$

this is assuming $$V^\pi \approx V^\pi_\psi$$.

Now we can write objective of policy gradient:

$$
\begin{align*}
&\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi(a \mid s_t^t) (R^\gamma (\tau_{t:T}) - V^\pi_\psi(s_t)) \\
&= \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi(a \mid s_t^t) (r(s_t,a_t) + \gamma V^\pi_\psi (s_{t+1}) - V^\pi_\psi (s_t)) \\
&= \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi(a \mid s_t^t) A(s_t,a_t)
\end{align*}
$$

where $$A(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi_\psi(s_t)$$ is the advantage. 

<b> Reminder </b> all of this is for calculating the policy gradient if you haven't been following along.

Using this calculation in our algorithm is called <b> Advantage Actor-Critic </b> (A2C).

The MC estimates of $$R^\gamma(\tau_{t:T})$$ has high randomness, while the randomness of $$r(s_t,a_t) + \gamma V^\pi_\psi(s_{t+1})$$ only comes from $$r(s_t,a_t)$$ since the value function is deterministic. This eliminates variance but introduces bias since $$V^\pi_\psi$$ is not the true value function and has underlying error.

Formally, we can write out the bias-variance trade-off. Consider an unbiased estimate $$h$$:

$$
\begin{equation*}
\text{Bias}(h) = \mathbb{E} [ X - h(X) ] = \mathbb{E} [ X ] - \mathbb{E} [ h(X) ] = 0
\end{equation*}
$$

where $$X$$ is the random variable and $$h(X)$$ is the estimate to $$X$$. In this case, $$X$$ is $$R^\gamma(\tau_{t:T})$$ and $$h(X)$$ is estimate of $$R^\gamma(\tau_{t:T})$$. Let's check out if MC estimation is unbiased estimate:

$$
\begin{equation*}
\mathbb{E} \left[ \frac{1}{N} \sum_{i=1}^N R^\gamma (\tau_{t:T}^i) \right] = \frac{1}{N} \sum_{i=1}^N \mathbb{E} [ R^\gamma (\tau_{t:T}^i)] = \frac{1}{N} \cdot N \mathbb{E} [ R^\gamma(\tau^i_{t:T})] = \mathbb{E} [ R^\gamma(\tau_{t:T})]
\end{equation*}
$$

The MC estimation method above is unbiased. 

Now let's check out our A2C method:

$$
\require{cancel}
\begin{equation*}
\mathbb{E} \left[ \frac{1}{N} \sum_{i=1}^N (r(s_t^i) + \gamma V^\pi_\psi) (s^i_{t+1}) \right] = \frac{1}{N} \sum_{i=1}^N \mathbb{E} [ r(s_t^i, a_t^i)] + \gamma \frac{1}{N} \mathbb{E} [ V^\pi_\psi (s^i_{t+1})] \cancel{=} \mathbb{E} [ R^\gamma (\tau_{t:T})]
\end{equation*}
$$

as we can see, there is no way A2C is unbiased. 

## 3.4.2 Generalized Advantage Estimation (GAE)

GAE balances bias and variance by mixing MC and approximated estimation. Consider $$r(s_t,a_t) + \gamma V^\pi_\psi (s_{t+1})$$ which is a biased estimate. One way to suppress variance in advantage estimation is incorporating more steps of rewards sampled from invronment. This is what we mean if we incorporate one more step:

$$
\begin{equation*}
A^1(s_t,a_t) = r(s_t,a_t) + \gamma r(s_{t+1}, a_{t+1}) + \gamma^2 V^\pi_\psi (s_{t+2}) - V^\pi_\psi (s_t)
\end{equation*}
$$

We can generalize this to $$n$$ steps:

$$
\begin{equation*}
A^n(s_t,a_t) = \sum_{i=0}^{n} \gamma^{i-1} r(s_{t+i}, a_{t+i}) + \gamma^{n+1} V^\pi_\psi (s_{t+n+1}) - V^\pi_\psi (s_t)
\end{equation*}
$$

How to choose $$n$$? Instead of picking, GAE uses exponential weighted average to mix all $$A^i$$ with a parameter $$\lambda$$ which gives us:

$$
\begin{equation}
A^{\text{GAE}(\gamma, \lambda)}(s_t, a_t) = \sum_{i=1}^T (\gamma \lambda)^i A^i(s_t, a_t)
\end{equation}
$$

where $$\lambda$$ controls bias and variance. If $$\lambda = 0$$, then we only use 1-step advantage estimation. If $$\lambda = 1$$, then we are doing MC estimation.

## 3.5 Exploration and Data Diversity

<b> Asynchronous sampling </b> A3C (Asynchronous Advantage Actor-Critic) is a variant of A2C that is asynchronous. It collects multiple trajectories in parallel using copies of the same policy. A3C consists of one master and multiple workers. This allows us to increase sample size.

Each worker fetches master's policy parameters asynchronously, computes policygradient using collected trajectories and sends gradient back to master thread. So what is the correctness of accumulating these gradients? Still leads to satisfactory performance.

<b> Entropy Regularization </b> Regularize policy gradients with entropy to increase diversity of actions executed by agent. Regularized objective is:

$$
\begin{align}
\sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi(a \mid s_t^i) A(s_t, a_t) + \mathcal{H}(\pi(a \mid s_t)) \\
\mathcal{H}(\pi(a \mid s_t)) = -\sum_{a \in \mathcal{A}} \pi(a \mid s_t) \log \pi(a \mid s_t)
\end{align}
$$

$$\mathcal{H}$$ is the entropy and we just take the entropy of the policy. This term encourages exploration by making parameters want to increase the entropy of the policy.

## 3.6 Conservative Policy Optimization

Now lets think about our policy gradient problem. Since our data collection comes from the same policy, a bad policy may collect bad data. It's a bad feedback loop where we may not be able to get a good policy. One way to get around this is to ensure our updates are conservative. We do this to make sure that we don't get into this bad feedback loop. This conservative update correlates to finding a conservative gradient step.

## 3.6.1 Parameter Space Constraint

One way to do this is to re-formulate our optimization problem:

$$
\begin{equation}
\max_{\theta} J(\theta) \\
\text{s.t.} \quad \| \theta \|_2^2 \leq \epsilon
\end{equation}
$$

We can re-cast this problem as:

$$
\begin{equation}
\max_{d} J(\theta + d) \\
\text{s.t.} \quad \| d \|_2^2 \leq \epsilon
\end{equation}
$$

we can get the Lagrangian of this problem:

$$
\begin{equation}
\max_{d} J(\theta + d) - \lambda \| d \|_2^2
\end{equation}
$$

Since $$J(\theta + d)$$ is unknown, we can use a first-order approximation:

$$
\begin{equation*}
J(\theta + d) \approx J(\theta) + d^T \nabla J(\theta) + O(d^2)
\end{equation*}
$$

Using this approximation, we can write the Lagrangian as:

$$
\begin{equation}
\max_{d} J(\theta) + d^T \nabla J(\theta) - \lambda \| d \|_2^2
\end{equation}
$$

if we set the gradient with respect to $$d$$ to zero, we get:

$$
\begin{align*}
0 &= \nabla J(\theta) - 2\lambda d \\
d &= \frac{1}{2\lambda} \nabla J(\theta)
\end{align*}
$$

which is just vanilla gradient ascent but we have a constraint on the step size or learning rate. Additionally, we want to make sure we follow this:

$$
\begin{equation*}
\| \beta d^* \|_2^2 \leq \epsilon
\end{equation*}
$$

which implies $$\beta = \sqrt{\frac{\epsilon}{\| d^* \|_2^2}}$$. And thus our gradient step is:

$$
\begin{equation}
\theta \gets \theta + \beta d^* = \theta + \sqrt{\frac{\epsilon}{\| d^* \|_2^2}} d^*
\end{equation}
$$

Another thing to note is how to pick $$\epsilon$$. We need to pick it such that the policy change is within a desired range. One thing to note is that the parameter difference depends on the current policy parametrizations and is not fixed.

## 3.6.2 Parametrization-independent Constraints on Output Space

This is where <b> Natural Policy Gradient (NPG) </b> comes in. We motivate NPG by looking at our constraints again:

$$
\begin{align*}
d^T d \leq \epsilon \\
\sum_{i,j} \mathbf{I}_{ij} d_i d_j \leq \epsilon
\end{align*}
$$

we want to say something like $$\| \partial \log \pi_\theta(a \mid s) \|_2^2 \leq \epsilon$$ rather than $$\| d \|_2^2 \leq \epsilon$$. This is more direct way of talking about small policy changes. We can do this by including the Fisher information matrix into the constraints above:

$$
\begin{align*}
d^T F d \leq \epsilon \\
\sum_{i,j} \frac{\partial \log \pi_\theta (a \mid s)}{\partial \theta_i} \frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_j} d_i d_j \leq \epsilon
\end{align*}
$$

where $$\mathbf{F}(\theta)$$ is the Fisher information matrix:

$$
\begin{equation}
\mathbf{F}(\theta) = \mathbb{E}_{\pi_\theta (a \mid s)} \left[ \nabla_\theta \log \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)^T \right]
\end{equation}
$$

which is the covariance of the gradient of the log-likelihood of the policy (expected value of outer product).


As $$\frac{\partial \log \pi_\theta (a \mid s)}{\partial \theta_i} \frac{\partial \log \pi_\theta(a \mid s)}{\partial \theta_j} d_i d_j$$ has unit where $$d_i = \partial \theta_i$$ and $$d_j = \partial \theta_j$$, we can see that:

$$
\begin{equation}
\frac{\partial \log \pi_\theta(a \mid s) \partial \log \pi_\theta(a \mid s) \partial \theta_i \partial \theta_j}
{\partial \theta_i \partial \theta_j} = 
(\partial \log \pi_\theta(a \mid s))^2
\end{equation}
$$

which makes the constraint pretty much $$ \sum_{i,j} (\partial \log \pi_\theta(a \mid s))^2 \leq \epsilon$$. This makes the constraint parametrization independent. Lots of rigorous analysis in information geometry to get this result. We just use this. Turns out Fisher information matrix is the only way to get parametrization independent metric.

Now we can write our final optimization problem:

$$
\begin{equation}
\max_{d} J(\theta + d) \\
\text{s.t.} \quad d^T \mathbf{F}(\theta) d \leq \epsilon
\end{equation}
$$

How do we solve this? We do something similar to before where we use first-order approximation:

$$
\begin{equation*}
J(\theta + d, \lambda) \approx J(\theta) + d^T \nabla J(\theta) - \lambda (\frac{1}{2} d^T \mathbf{F}(\theta) d - \epsilon)
\end{equation*}
$$

then set gradient to zero and solve:

$$
\begin{align*}
0 &= \nabla J(\theta) - \lambda \mathbf{F}(\theta) d \\
d^* &= \frac{1}{\lambda} \mathbf{F}(\theta)^{-1} \nabla J(\theta)
\end{align*}
$$

Now we just need to tighten the constraint $$d^T \mathbf{F}(\theta) d \leq \epsilon$$ just like before using $$\beta$$. We do this by plugging $$d^*$$ into constriant:

$$
\begin{align*}
&\frac{\lambda^2}{2} (d^*)^T \mathbf{F}(\theta) d^* \\
&= \frac{\lambda^2}{2} (\beta \mathbf{F}(\theta)^{-1} \nabla J(\theta))^T \mathbf{F}(\theta) (\beta \mathbf{F}(\theta)^{-1} \nabla J(\theta)) \\
&= \frac{\lambda^2 \beta^2}{2} \nabla J(\theta)^T \mathbf{F}(\theta)^{-T} \mathbf{F}(\theta) \mathbf{F}(\theta)^{-1} \nabla J(\theta) \\
&= \frac{\lambda^2 \beta^2}{2} \nabla J(\theta)^T \mathbf{F}(\theta)^{-1} \nabla J(\theta) \leq \epsilon
\end{align*}
$$

we use knowledge that $$\mathbf{F}(\theta)^{-T} = \mathbf{F}(\theta)^{-1}$$. Now we can solve for $$\beta$$ to find the value that tightens inequality:

$$
\begin{equation}
\beta^* = \frac{1}{\lambda} \sqrt{\frac{2\epsilon}{\nabla J(\theta)^T \mathbf{F}(\theta)^{-1} \nabla J(\theta)}}
\end{equation}
$$

same as before, we can plug this into our gradient step:$$\theta \gets \theta + \beta^* d^*$$

<b> Connection to Kullback-Leibler (KL) Divergence </b> The second-order expansion of KL-divergence is Fisher information matrix, thus NPG can be rewritten as KL-divergence:

$$
\begin{equation}
\max_{d} J(\theta + d) \\
\text{s.t.} \quad D_{KL}(\pi_\theta(a \mid s) || \pi_{\theta + d}(a \mid s)) \leq \epsilon
\end{equation}
$$

<b> Connection to Newton Method </b> NPG is very similar to Newton's method for second-order optimizaiton:

$$
\begin{align*}
\theta &\gets \theta + \beta \mathbf{H}^{-1}(\theta) \nabla J(\theta) \\
\theta &\gets \theta + \beta \mathbf{F}(\theta)^{-1} \nabla J(\theta)
\end{align*}
$$

where the first expression is Newton's method ($$\mathbf{H}(\theta)$$ is hessian) and second expression is NPG. Since we know that Fisher information matrix is the second-order approximation of KL-divergence, we can see that NPG is Newton's method of the KL-divergence problem.

## 3.6.3 Monotonic Policy Improvement

While NPG ensures change in policy bounded by $$\epsilon$$, it does not guarantee that each policy update will improve policy performance measured by returns. <b> Trust Region Policy Optimization </b> (TRPO) mitigates this by maximizing policy improvement instead of just constraining changes in policy outputs. If policy improvement is guaranteed, then agent will not get stuck in poor local maxima and avoid the bad feedback loop (vicious cycle). 

We describe the problem as:

$$
\begin{equation}
\max_\theta J(\theta + d) - J(\theta) 
\end{equation}
$$

where the difference quantifies policy improvement of $$\pi_{\theta + d}$$ with respect to old policy $$\pi_\theta$$. We can show policy improvement can be rewritten as:

$$
\begin{equation*}
J(\theta + d) - J(\theta) = \mathbb{E}_{\pi \sim \pi_{\theta + d}} \left[ \sum_{t=1}^T \gamma^{t-1} A^{\pi_\theta}(s_t,a_t) \right]
\end{equation*}
$$

with this in mind, we restate optimization as:

$$
\begin{equation*}
\max_d \mathbb{E}_{\tau \sim \pi_{\theta + d}} \left[ \sum_{t=1}^T \gamma^{t-1} A^{\pi_\theta}(s_t,a_t) \right]
\end{equation*}
$$

We can break down this equation as follows:

$$
\begin{align*}
\mathbb{E}_{\tau \sim \pi_{\theta + d}} \left[ \sum_{t} \gamma^{t-1} A^{\pi_\theta}(s_t,a_t) \right] &= \sum_{s,a} \sum_t P(s=s_t,a=a_t \mid \pi_{\theta + d})[\gamma^{t-1} A^{\pi_\theta}(s,a)] \\
&= \sum_t \sum_{s,a} \gamma^{t-1} P(s=s_t,a=a_t \mid \pi_{\theta + d}) [A^{\pi_\theta}(s,a)] \\
&= \sum_t \sum_s \gamma^{t-1} P(s_t=s \mid \pi_{\theta + d}) \sum_a \pi_{\theta + d}(a_t = a \mid s_t) [ A^{\pi_\theta}(s,a) ] \\
&= \sum_s \rho^\gamma_{\pi_{\theta + d}}(s) \sum_a \pi_{\theta + d} (a \mid s) [A^{\pi_\theta}(s,a)] \\
&= \sum_s \rho^\gamma_{\pi_{\theta + d}}(s) \sum_s \pi_\theta(a \mid s) \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta(a \mid s)} [ A^{\pi_\theta}(s,a) ] \\
&= \mathbb{E}_{s \sim \pi_{\theta + d}, a \sim \pi_\theta} \left[ \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s,a) \right]
\end{align*}
$$

where $$\rho^\gamma_{\pi_{\theta + d}}(s) = \sum_t \gamma^{t-1} P(s=s_t \mid \pi_{\theta + d})$$.

We can then rewrite the optimization problem as:

$$
\begin{equation}
\max_d \mathbb{E}_{s \sim \pi_{\theta + d}, a \sim \pi_\theta} \left[ \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s,a) \right]
\end{equation}
$$

since it requires sampling from the new policy $$\pi_{\theta + d}$$, there is an inefficiency there (need rollouts in environment with new policy). We can approximate the performance of the new policy using state distribution of old policy $$\pi_\theta$$:

$$
\begin{equation}
J(\theta + d) \geq \hat J_\theta (\theta + d) - C \cdot D_{KL}^{\max}(\pi_\theta || \pi_{\theta + d}(a \mid s))
\end{equation}
$$

where $$C = \frac{4 \epsilon \gamma}{(1-\gamma)^2}$$, $$D^{\max}_{KL} = \max_s D_{KL}$$, and $$\hat J(\theta + d) = \mathbb{E}_{s \sim \pi_\theta, a \sim \pi_{\theta + d}} [A^{\pi_\theta}(s,a)]$$.

<b> KL Divergence </b>:

$$
\begin{align}
D_{KL}(P || Q) &= \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)} \\
&= \mathbb{E}_{x \sim P} \left[ \log\frac{p(x)}{q(x)} \right]
\end{align}
$$

Using this new approximation, we can rewrite the optimization problem as:

$$
\begin{equation}
\max_d \mathbb{E}_{s \sim \pi_\theta, a \sim_\theta} \left[ \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta (a \mid s)}A^{\pi_\theta}(s,a) \right] - C D_{KL}^\max(\pi_\theta(a \mid s) || \pi_{\theta + d}(a \mid s))
\end{equation}
$$

This is TRPO. The approximation we're using at the end here is only valid if the state distributions from the new policy is similar to the old policy. However, how small of a change we need to make for this to be valid is in practice very small. (especially if we use $$C$$'s derived expression).

Another issue is that $$D_{KL}^\max$$ is typically intractable (I've only ever calculated this for gaussians. We don't want to only have gaussian policies). Instead we can use the expected KL-divergence (which is always estimated through sampling). Then we move the KL-divergence term as a constraint to enable larger step sizes:

$$
\begin{equation}
\max_d \mathbb{E}_{s \sim \pi_\theta, a \sim_\theta} \left[ \frac{\pi_\theta(a \mid s)}{\pi_\theta(a \mid s)} A^{\pi_\theta}(s,a) \right] \\
\text{s.t.} \quad \mathbb{E}_s \left[ D_{KL}(\pi_\theta(a \mid s) || \pi_{\theta + d}(a \mid s)) \right] \leq \epsilon
\end{equation}
$$

This looks awfully similar to the NPG problem we had before. Turns out we use the same methodology to solve this problem and get gradient update steps. There is an extra step employed which uses conjugate gradients instead to avoid calculating the matrix inverse of the hessian. What's the difference? Before, we were using the Fisher information matrix to get the gradient step which doesn't guarantee performance improvement. Now, we are using the KL-divergence to get the gradient step which "guarantees" performance improvement given small enough steps.

With all of these approximations, how do we know that our policy will improve? We will have to use $$\beta$$ to make this guaranteed and the constraint is tightened/satisfied. We do the following:

$$
\beta \gets \begin{cases}
0.5 \beta  & \text{if } D_{KL}(\pi_\theta(a\mid s) || \pi_{\theta + d}(a \mid s)) < \epsilon \\
0.5 \beta  & \text{if } \hat J(\theta, d_{\text{new}}) - \hat J(\theta, d_{\text{old}}) < 0 \\
\beta & \text{otherwise}
\end{cases}
$$

At first, $$\beta$$ is initialized using NPG's method.

<b> Adaptive Lagrangian Method </b> Line search may be expensive, we can solve the unconstrained lagrangian and use adaptive updates on the lagrange multiplier. 

$$
\begin{equation}
\max_d \mathbb{E}_{s \sim \pi_\theta, a \sim \pi_\theta} \left[ \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta (a \mid s)} A^{\pi_\theta}(s,a) - \lambda D_{KL}(\pi_\theta(\cdot \mid s) || \pi_{\theta + d}(\cdot \mid s)) \right]
\end{equation}
$$

these are updates for lagrange multiplier:

$$
\begin{equation}
\lambda \gets \begin{cases}
\frac{\lambda}{2} & \delta_{KL} < \delta_{\text{target}} / 1.5 \\
2\lambda & \delta_{KL} \geq \delta_{\text{target}} \cdot 1.5
\end{cases}
\end{equation}
$$

where $$\delta_{KL} = D_{KL}(\pi_\theta(a\mid s) \| \pi_{\theta + d}(a \mid s))$$ and $$\delta_{\text{target}}$$ is arbitrary threshold.

<b> Clipping </b> Removing the KL-divergence constraint, we can propose to clip the objective function instead of enforcing the constraint in policy updates. This is called Proximal Policy Optimization (PPO). To figure this out, we restart back to the original optimization problem (still doing monotonic policy improvement):

$$
\begin{equation*}
\mathbb{E}_{s \sim \pi_{\theta + d}, a \sim \pi_\theta} \left[ \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta (a \mid s)} A^{\pi_\theta}(s,a) \right]
\end{equation*}
$$

as $$\frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta (a \mid s)}$$ weighs the advantage, the idea of PPO is to penalize a new policy $$\pi_{\theta + d}$$ that has either a large or small ratio $$c_t = \frac{\pi_{\theta + d}(a \mid s)}{\pi_\theta (a \mid s)}$$. This penalization is realized by clipping $$c_t$$ within a pre-defined bound:

$$
\begin{equation}
\mathbb{E}_{s \sim \pi_\theta, a \sim \pi_\theta} \left[ \min \{ \frac{\pi_{\theta+d}(a\mid s)}{\pi_\theta (a \mid s)}A^{\pi_\theta}(s,a), \text{clip}(\frac{\pi_{\theta+d}(a\mid s)}{\pi_\theta (a \mid s)}, 1-\epsilon, 1+\epsilon) A^{\pi_\theta}(s,a) \} \right]
\end{equation}
$$

so we choose between the clipped and unclipped objective (whichever one is better). This has way less compute and performs better than TRPO.

## 3.7 Practical Considerations of PPO

PPO uses a lot of code-level optimzations to make it work. Without these optimizations, PPO cannot outperform TRPO. This is one of the reasons why PPO has such a wide range of performance on the same tasks across open-sourced implementations.

Read [this](https://arxiv.org/pdf/2005.12729) for more details on the practical considerations of PPO.

## 3.8 Debugging

- Visualize policy performance at initialization
- Analyze returns at initialization
- Increase batch size
- Periodically visualize agent behavior
- Entropy of actions