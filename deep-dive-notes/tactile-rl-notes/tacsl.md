---
layout: default
title: TacSL
parent: Tactile RL Notes
mathjax: true
tags: 
  - latex
  - math
has_children: true
---

# TacSL

Equations they use for simulating shear fields is:

For each point on the elastomer, we calculate the contact force acting on it due to object penetration.
$$
\begin{align}
\mathbf{f}_n &= (-k_n + k_d \dot d) d \cdot \mathbf{n} \\
\mathbf{f}_t &= -\frac{\mathbf{v}_t}{\|\mathbf{v}_t\|} \min\{k_t \| \mathbf{v}_t\|, \mu \|\mathbf{f}_n\|\}
\end{align}
$$
- $$k_n, k_d$$ are spring/damping constants.
- $$d$$: penetration depth of the point (gotten from object-elastomer sdf calculation).
- $$\mathbf{v}_t$$: tangential contact velocity.
- they  set $$k_d = 0$$
- for $$-\frac{\mathbf{v}_t}{\|\mathbf{v}_t\|}$$, they clamp the minimum of the norm to avoid NaN.