---
layout: default
title:  "GNN as Gradient Flow"
date:   2025-04-17
categories: GNN Diffusion
parent: Quick Notes on Graphs
# math: katex
mathjax: true
tags: 
  - latex
  - math
---

# Graph Neural Networks as gradient flows

This is kind of diving deeper into a physics-inspired architecture of the GNN Diffusion perspective. 

### Gradient Flows
Consider a dynamical system governed by evolution equation:

$$
\begin{equation}
\dot{\mathbf{X}} = \mathbf{F}(\mathbf{X}(t))
\end{equation}
$$

Gradient Flows are special types of evolution equations of the form

$$
\begin{equation}
\mathbf{F(\mathbf{X}(t))} = -\nabla \mathcal{E}(\mathbf{X}(t))
\end{equation}
$$

where $$\mathcal{E}$$ is an energy functional. Gradient flow makes $$\mathcal{E}$$ monotonically decrease during evolution.

### Diffusion Equation

The diffusion equation is given by:

$$
\begin{equation}
\dot{\mathbf{X}}(t) = \Delta \mathbf{X}
\end{equation}
$$

which is an example of gradient flow. $$\Delta$$ is an nxn graph laplacian matrix.

The diffusion equation is the gradient flow of the Dirichlet Energy:

$$
\begin{align}
\mathcal{E}^{\text{DIR}}(\mathbf{X}) &= \frac{1}{2} \text{trace}(\mathbf{X}^T \Delta \mathbf{X}) \\
&= \frac{1}{2} \sum_{uv} \| (\nabla \mathbf{X})_{uv} \|^2
\end{align}
$$

where $$(\nabla \mathbf{X})_{uv}$$ is the gradient of the features on every edge of the graphl The Dirichlet energy measures the smoothness of the features on the graph. As $$t \to \infty$$, the features will converge to a vector and all become the same (oversmoothing).

### Convolutional GNN

The current diffusion equation is not too useful in Graph ML. We should do an interpretation of the equation that is most relevant to what we'll use in practice (GNNs). Let this be our evolution equation:

$$
\begin{equation}
\dot{\mathbf{X}}(t) = \text{GNN}(\mathcal{G}, \mathbf{X}(t) ; \theta(t))
\end{equation}
$$

which we discretize with forward-euler using timestep $$0 < \tau < 1$$. 

$$
\begin{equation}
\mathbf{X}(t + \tau) = \mathbf{X}(t) + \tau \text{GNN}(\mathcal{G}, \mathbf{X}(t), \theta(t))
\end{equation}
$$

This is how we can describe GNN message-passing in terms of gradient flow. Say we were doing GCNs, we can write the GNN as:

$$
\begin{equation}
\mathbf{X}(t + \tau) = \mathbf{X}(t) + \tau \sigma(-\mathbf{X}(t) \Omega(t) + \tilde{\mathbf{A}}\mathbf{X}(t)\mathbf{W}(t))
\end{equation}
$$

where $$\Omega, \mathbf{W}$$ are learnable matrices and $$\tilde{\mathbf{A}}$$ is the normalized adjacency matrix. Simplest convolutional graph network is when $$\Omega = 0$$.

$$
\begin{equation}
\end{equation}
$$

### Parametrising Energy

Instead of parametrising the evolution equation, we parametrise the energy and utilize its gradient flow as the evolution equation:

$$
\begin{equation}
\dot{\mathbf{X}}(t) = - \nabla \mathcal{E}(\mathcal{G}, \mathbf{X}(t) ; \theta(t))
\end{equation}
$$

of some energy $$\mathcal{E}$$. Now we can study families of energy functions to find out which ones are effective to create evolution equations for our GNNs.

### Energy to Evolution

Consider the family of quadratic energy equations:

$$
\begin{equation}
\mathcal{E}^{\theta}(\mathbf{X}) = \frac{1}{2} \sum_{u}\langle \mathbf{x}_u, \Omega \mathbf{x}_u \rangle - \frac{1}{2} \sum_{uv} \bar{a}_{uv} \langle \mathbf{x}_u, \mathbf{W} \mathbf{x}_v \rangle
\end{equation}
$$

where we parametrize $$\Omega, \mathbf{W}$$. The first term is external energy on all particles. The second term is pair-wise interactions along edges of the graph (internal energy). Minimizing the second term makes the each term $$\mathbf{x}_u, \mathbf{x}_v$$ of adjacent nodes attract along the positive eigenvectors of $$\mathbf{W}$$. Negative eigenvectors repel.

The gradient flow of this is given by the following (also assume $$\mathbf{W},\Omega$$ are symmetric):

$$
\begin{align}
\dot{\mathbf{X}}(t) &= -\nabla \mathcal{E}^{\theta}(\mathbf{X}(t)) \\
&= -\mathbf{X}(t) \frac{(\Omega + \Omega^T)}{2} + \tilde{A}\mathbf{X}(t)\frac{(\mathbf{W} + \mathbf{W}^T)}{2} \\
&= -\mathbf{X}(t)\Omega + \tilde{A}\mathbf{X}(t)\mathbf{W}
\end{align}
$$

Then when we use forward-euler, we get the following:

$$
\begin{equation}
\mathbf{X}(t + \tau) = \mathbf{X}(t) + \tau \left( -\mathbf{X}(t)\Omega + \tilde{A}\mathbf{X}(t)\mathbf{W} \right)
\end{equation}
$$

### Dominant Effects in dynamics

We can interpret the dynamics of the layer-system by analyzing the Dirichlet energy function of the system. Attractive forces minimise the edge gradients which produces smoothing effects (blurring) that magnify low frequency signals of the node features while repulsive forces increase edge gradients and produce a sharpening effect.

We can analyze the Dirichlet energy function to see this by checking if in the limit $$\mathcal{E}^{\text{DIR}}(\frac{\mathbf{X}(t)}{\| \mathbf{X}(t) \|})$$ tends to 0. We would have low-frequency dominated (LFD) dynamics. In the case of convergence to the largest eigenvalue, we call the dynamics high-frequency dominated (HFD) dynamics.

Many studies have shown LFD systems do well in node classification while HFD systems do well for heterophilic tasks.

The previous works (GRAND, PDE-GCN-D, Continuous-GNN) are unable to induce HFD dynamics and will suffer in heterophilic cases. However, through our current modelling (named GRAFF), we can learn on heterophilic settings.