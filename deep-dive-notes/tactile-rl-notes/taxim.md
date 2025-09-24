---
layout: default
title: Taxim
parent: Tactile RL Notes
mathjax: true
tags: 
  - latex
  - math
has_children: true
---

# Taxim

Taxim uses a reflection function inspired by photometric stereo to model tactile rgb rendering.

Photometric stereo linear reflection function is defined by:

$$
\begin{align}
I = \sum_l \alpha_l \mathbf{n}
\end{align}
$$

where $$I$$ is the intensity (we can have $$I_r, I_g, I_b$$ for all rgb values).

However, we can note that this is linear and the lighting is not necessarily linear in the environment, so we go for a more nonlinear function:


We are given a background image $$I_0$$ (image of gel without contact). We want to find the image while in contact $$I_1$$. we do this by finding the change in the image $$\Delta I$$.

$$
\begin{align}
I_1 &= I_0 + \Delta I \\
\Delta I &= (\mathbf{w}_n)^T
\begin{bmatrix}
x^2 & y^2 & xy & x & y & 1
\end{bmatrix}^T
\end{align}
$$
- $$\Delta I$$: intensity at location (x,y) in pixel coordinates.
- We'll have $$I_r, I_g, I_b$$ (3 sets of weights).
- $$\mathbf{w}_n$$: weights for light source $$l$$ and normal direction $$n$$.
- normal direction $$n$$ is discretized into 125x125 bins (binning by direction in spherical coordinates).


where $$\mathbf{w}^l_n$$ is a list of weights for a 2nd degree polynomial.

Note that the subscript $$n$$ denotes the fact that the weights are looked up based on the normal direction.

