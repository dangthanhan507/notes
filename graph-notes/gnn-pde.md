---
layout: default
title:  "GNNs as PDEs"
date:   2025-04-04
categories: GNN and PDE
parent: Quick Notes on Graphs
# math: katex
mathjax: true
tags: 
  - latex
  - math
---

# Graph Neural Diffusion

This is based on [Graph Neural Diffusion](https://arxiv.org/abs/2106.10934) (GND).

## Preliminary on Diffusion

Isaac Newton anonymously published "A Scale of the Degrees of Heat" in 1701. He proposed that heat diffuses through a medium, and the rate of diffusion is proportional to the temperature gradient. A modern phrasing states, "the temperature a hot body loses in a given time is proportional to the temperature difference between the object and the environment". This gives rise to the heat diffusion equation:

$$
\begin{equation}
\dot x = \alpha \Delta x
\end{equation}
$$

$$x(u,t)$$ is the temperature at time t and point u. The $$\Delta$$ is the Laplacian operator. It expresses local difference between temperature of point and its surroundings. The $$\alpha$$ is the thermal diffusivity constant.

This PDE is linear and has a closed form solution. The solution is given by the convolution of the initial condition with a Gaussian kernel:

$$
\begin{equation}
x(u,t) = x(u,0) \text{exp}^{-\frac{\lvert u \rvert}{4t}}
\end{equation}
$$

A more general form of the diffusion equation (Fourier's heat transfer law) is:

$$
\begin{equation}
\dot x(u,t) = div(\alpha(u,t) \nabla x(u,t))
\end{equation}
$$

---

## Diffusion PDEs

Diffusion PDEs arise in physical processes that transfer energy/matter/information. An image processing perspective is to interpret diffusion as a linear low-pass filter for image denoising. However, this filter for image denoising gives a blurring effect on areas with high color gradient (contrast). The bilateral filtering paper from Jitendra Malik's group proposed using an adaptive diffusivity coefficient inversely dependent on the norm of the image gradient. Diffusion is strong in "flat" regions and weak in high gradient regions (Perona-Malik diffusion).

[Perona-Malik diffusion](/home/an-dang/anisotropic_diffusion.pdf) created an entire [field](https://www.mia.uni-saarland.de/weickert/Papers/book.pdf) of PDE-based techniques that drew inspiration and methods from geometry, calculus of variations, and numerical analysis. Bronstein was inspired to do differential geometry because of Ron Kimmel's [work](https://books.google.co.uk/books?id=su7xBwAAQBAJ) on numerical geometry of images. Variational and PDE-based methods were widely used until 2020s when deep learning took over.

---

## GRAND Takeaway

Bronstein's work on GRAND (Grand Neural Diffusion) takes a similar philosophy. GNNs operate by exchanging information between nodes through message-passing. The message-passing process can be interpreted as a diffusion process on the graph. The diffusion processes to a graph $$\mathcal{G} = (\mathcal{V}, \mathcal{E})$$ looks like this:

$$
\begin{equation}
\mathbf{\dot X}(t) = div(\mathbf{A}(\mathbf{X}(t)) \nabla \mathbf{X}(t))
\end{equation}
$$

where $$\mathbf{X}(t)$$ is a $$\lvert \mathcal{V} \rvert \times n_f$$ matrix of node features at time t. $$(\nabla \mathbf{X})_{uv} = \mathbf{x}_v - \mathbf{x}_u$$ is the gradient. The divergence is defined as $$(\text{div}(\mathbf{X}(t)))_u = \sum_{v \in \mathcal{N}(u)} \mathbf{w}_{uv} \mathbf{x}_{uv}$$ where $$\mathbf{w}_{uv}$$ is the edge feature between nodes u and v. Diffusivity is defined as $$\mathbf{A}(\mathbf{X}(t)) = \text{diag}(\alpha(\mathbf{x}_u, \mathbf{x}_v))

Now with all of this, the final graph diffusion equation is:

$$
\begin{equation}
\dot{\mathbf{X}}(t) = (\mathbf{A}(\mathbf{X}(t)) - \mathbf{I})\mathbf{X}(t)
\end{equation}
$$

In most cases, this has no closed-form solution (needs to be solved numerically). 


## Integration of PDE Equation
We can simply start a forward time difference discretization of the diffusion PDE:

$$
\begin{align}
\frac{[\mathbf{X}(k+1) - \mathbf{X}(k)]}{\tau} &= (\mathbf{A}(\mathbf{X}(k)) - \mathbf{I})\mathbf{X}(k) \\
\mathbf{X}(k+1) - \mathbf{X}(k) &= \tau(\mathbf{A}(\mathbf{X}(k)) - \mathbf{I})\mathbf{X}(k) \\
\mathbf{X}(k+1) &= \tau(\mathbf{A}(\mathbf{X}(k)) - \mathbf{I})\mathbf{X}(k) + \mathbf{X}(k) \\
\mathbf{X}(k+1) &= \tau(\mathbf{A}(\mathbf{X}(k)) - \frac{1-\tau}{\tau}\mathbf{I})\mathbf{X}(k) \\
\mathbf{X}(k+1) &= \mathbf{Q}(k)\mathbf{X}(k)
\end{align}
$$

If we tried the backward time difference discretization, we would get:

$$
\begin{align}
[(1+\tau)\mathbf{I} - \tau \mathbf{A}(\mathbf{X}(k))]\mathbf{X}(k+1) &= \mathbf{X}(k) \\
\mathbf{B}(k)\mathbf{X}(k+1) &= \mathbf{X}(k) \\
\mathbf{X}(k+1) &= \mathbf{B}^{-1}(k)\mathbf{X}(k)
\end{align}
$$

This is a semi-implicit scheme where we have to solve the inverse of $$\mathbf{B}$$ to get the next state. These are overall not the best in practice. Using Runge-Kutta methods are better.

However, if you don't know about higher-order multi-step methods, then stick with the Euler methods.

---

## Connection to current GNNs

We can look at the GNN architectures as a discretized instance of Graph Diffusion. Equation 10 is just Graph Attention Transformer where $$\mathbf{A}$$ is the attention. These GNNs use explicit single-step Euler scheme. You can think of diffusion stepping as a message-passing step. Adaptive time-stepping can allow us to use fewer GNN layers and message passing steps if we consider the continuous interpretation of GNN message-passing.

Additionally, graph rewiring (spawned from this work) sis popular to address over-smoothing or bottlenecks. The diffusion framework offers a principled view on graph rewiring by considering the graph as a spatial discretization of some continuous object.

---

## GRAND (paper)