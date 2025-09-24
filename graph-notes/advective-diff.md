---
layout: default
title:  "Advective Diffusion"
date:   2025-04-17
categories: Topological Generalization
parent: Quick Notes on Graphs
# math: katex
mathjax: true
tags: 
  - latex
  - math
---

# Topological Generalisation with Advective Diffusion Transformers

This is a continuation of my notes from the graph diffusion perspective. Advective diffusion transformers (diffusion through message-passing) combines message-passing with transformers for better topological generalisation.

GNNS are often reported to show poor performance when train/test datasets are generated from different distributions (ditribution shift... or topological shift).

### Neural Graph Diffusion

Neural graph diffusion (NGD) formulates a continuous-time differential equation of the GNN layers using the message-passing connection with heat diffusion equation.

$$
\begin{align}
\dot{\mathbf{X}}(t) = \text{div}(\mathbf{S}(t) \nabla \mathbf{X}(t))
\end{align}
$$

We initialize this equation with

$$
\begin{align}
\mathbf{X}(0) = \text{ENC}(\mathbf{X}_i)
\end{align}
$$

Then we run the differential equation and integrate it to get the following.

$$
\begin{align}
\mathbf{X}_o = \text{DEC}(\mathbf{X}(T))
\end{align}
$$

which is then the output of the architecture.

We can now interpret most message-passing layers as some form of the stated diffusion equation. For example, the linear version of this equation is the GCN layer. The nonlinear version is the GAT layer. The generalisation ability of the NGD is up to the sensitivity of the solution $$\mathbf{X}(T) = f(\mathbf{X}(0), \mathbf{A})$$ to a perturbation of the adjacency matrix $$\tilde{\mathbf{A}} = \mathbf{A} + \delta \mathbf{A}$$. For both nonlinear and linear graph diffusion, the solution is still highly sensitive to adjacency perturbations.

### Graph Transformers

So the graph transformer processes the graph using a computational method much closer to the actual QKV attention mechanism described:

$$
\begin{equation}
\text{Att}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}
\end{equation}
$$

For the graph transformer, we calculate the following:
$$
\begin{align}
\mathbf{Q} &= \mathbf{X} \mathbf{W}_Q \\
\mathbf{K} &= \mathbf{X} \mathbf{W}_K \\
\mathbf{V} &= \mathbf{X} \mathbf{W}_V
\end{align}
$$

We then re-use the attention equation to calculate the attention scores for every node in the graph. If there are edge features, we can use those to modify the attention scores. However, it is important to note that this is done GLOBALLY on a graph instead of GAT which is done locally on a single node's neighbors for every node.

This can also be seen as non-local diffusion in our context. This non-local diffusion gives stronger topological generalisation. However this asssumes that node labels and graph topology are statistically independent. However, we want to analyze this where they are statistically dependent.


### Advective Graph Diffusion

We now modify our diffusion equation to contain an advective term:

$$
\begin{equation}
\dot{\mathbf{X}}(t) = \text{div}(\mathbf{S}(t) \nabla \mathbf{X}(t)) + \beta\text{div}(\mathbf{V}(t) \nabla \mathbf{X}(t))
\end{equation}
$$

this is known as advective diffusion which arises in fluid dynamics. The advective term relates to the movement of the water.

The diffusion term comes from the graph transformer term (global attention) and the advection term comes from local operations. 

We will use the equation above similarly to how NGD uses its diffusion equation.

This was shown to be superior to the previous models (GCN, GAT, MPNN, NGD) in terms of topological generalisation for the task of graph generation.