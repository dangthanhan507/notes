---
layout: default
title: 2. Flow
parent: Unsupervised Learning
nav_order: 3
mathjax: true
tags: 
  - latex
  - math
# math: katex
---

# 2. Likelihood Models: Flow Models

We stil want to learn a distribution $$p_\theta(x)$$ and sample from it.

Recall Autoregressive models issues:
- Sampling is serial (hence slow)
- Lacks latent representation / embedding space
- Limited to discrete data (at least in terms of experimental success)

Flow models will address these issues, but its performance not as good as other models.

## 2.1 Foundation of Flows

<b> Probability Density Function (PDF) </b>: $$ P( x \in [a,b] ) = \int_a^b p(x) dx$$ this is probability that x lands in an interval.

<b> Review: Fitting density model </b>: We formulate max likelihood problem:

$$
\begin{equation}
\max_\theta \sum_i \log p_\theta(x^{(i)})
\end{equation}
$$

where $$x^{(i)}$$ is the i-th data point. We can reformulate this with SGD and minimization:

$$
\begin{equation}
\min_\theta \mathbb{E}_{x \sim p_\theta} \left[ -\log p_\theta(x) \right]
\end{equation}
$$

Another way to quantize (discretize) the data to a discrete distribution instead of continuous.

<b> Example Density Model </b>: Mixtures of Gaussians (MoG):

$$
\begin{equation}
p_\theta (x) = \sum_{i=1}^k \pi_i \mathcal{N}(x;\mu_i, \sigma_i^2)
\end{equation}
$$

where $$\theta = \{ \pi_1, ..., \pi_k, \mu_1, ..., \mu_k, \sigma_1^2, ..., \sigma_k^2 \}$$. Our goal is to learn the parameters with NLL loss.

Gaussian Perturbation of an image will lead to an unlikely image. Nearby an image, there is not many correct images. 
- What if we tried other parametrized distributions? Not that many choices. We admit defeat and want to find a better fit.

Flow Models refer to us going from embedding space back to data space. 

Say $$ z = f_\theta (x) $$. We wish to sample by getting $$x = f_\theta^{-1}(z)$$.

This assumes that $$f$$ has an inverse. This is a strong assumption. VAE is where we learn the inverse instead. 

$$
\begin{align*}
z = f_\theta (x) \\ 
p_\theta (x) dx = p(z) dz \\
p_\theta (x) = p(f_\theta (x)) \left| \frac{\partial f_\theta (x)}{\partial x} \right|
\end{align*}
$$

$$f$$ needs to be differentiable and invertible.

Now, we can view this in terms of datapoint $$x^{(i)}$$.

$$
\begin{align*}
p_\theta (x^{(i)}) &= p_Z(z^{(i)}) \left| \frac{\partial z}{\partial x}(x^{(i)}) \right| \\
&= p_Z(f_\theta (x^{(i)})) \left| \frac{\partial f_\theta }{\partial x}(x^{(i)}) \right|
\end{align*}
$$

Our maximum likelihood objective is:

$$
\begin{equation}
\max_\theta \sum_i \log p_\theta(x^{(i)}) = \max_\theta \sum_i \log p_Z(f_\theta (x^{(i)})) + \log \left| \frac{\partial f_\theta }{\partial x}(x^{(i)}) \right| \\
\end{equation}
$$

Choices to make:
1. Invertible function class.
  - exp
  - polynomials
  - cumulative distribution functions (CDFs)... mixture of gaussians
  - compose flows
2. Embedding space density
  - ideally easy distribution to sample from
  - unit gaussian (normalizing flows)
  - mixture of gaussians
  - uniform distribution

How to sample from $$p_\theta(x)$$? First, sample from $$p_Z(z)$$. Then, apply the inverse function $$f_\theta^{-1}$$ to get back to data space.

Remember that we can use inverse CDF to sample from any distribution. We start from uniform distribution ($$p_Z$$) and apply the inverse CDF to sample distribution that we want.

Really how we have been doing inverse CDF is a special case of flow. 

## 2.2 2D Flows

We have the following:

$$ x_1 \mapsto z_1 = f_{\theta_1}(x_1)$$

$$ z_1 \mapsto x_1 = f_{\theta_1}^{-1}(z_1)$$

$$ x_2 \mapsto z_2 = f_{\theta_2}(x_1,x_2)$$

$$ z_2,x_1 \mapsto x_2 = f_{\theta_2}^{-1}(x_1,z_2)$$

If we do:

$$ z_2,x_1 \mapsto x_2 = f_{\theta_2}^{-1}(z_1,z_2)$$

then this is inverse autoregressive flow.

Remember, this is for 2-D flow which is just a 2-dimensional vector that we are sampling. 

### 2.2.1 Training Objective

$$
\begin{align*}
\log p_\theta(x_1, x_2) &= \log p_{Z_1}(z_1) + \log \left| \frac{\partial z_1 (x_1)}{\partial x_1} \right| \\
&+ \log p_{Z_2}(z_2) + \log \left| \frac{\partial z_2 (x_1, x_2)}{\partial x_2} \right| \\
&= \log p_{Z_1}(f_{\theta_1}(x_1)) + \log \left| \frac{\partial f_{\theta_1}(x_1)}{\partial x_1} \right| \\
&+ \log p_{Z_2}(f_{\theta_2}(x_1,x_2)) + \log \left| \frac{\partial f_{\theta_2}(x_1,x_2)}{\partial x_2} \right|
\end{align*}
$$

Then use this with NLL to train the model.

Data distribution will be a set of 2d vectors.
We try to learn the distribution of 2d vectors using this format.

## 2.3 N-D Flows

We can see from the previous section the obvious autoregressive extension for n-dimensional flows.

$$
\begin{align*}
x_1 \sim p_\theta(x_1) \\
x_1 = f_{\theta}^{-1}(z_1) \\
x_2 \sum p_\theta(x_2|x_1) \\
x_2 = f_{\theta}^{-1}(z_2;x_1) \\
x_3 \sim p_\theta(x_3|x_1,x_2) \\
x_3 = f_{\theta}^{-1}(z_3;x_1,x_2)
\end{align*}
$$

Sampling is invertible mapping from $$z$$ to $$x$$.

We train as before.

<b> Inverse Autoregressive Flow </b>:
## 2.4 Dequantization

