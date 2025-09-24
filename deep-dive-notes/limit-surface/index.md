---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Limit Surface Planner
parent: Deep Dive Ideas
mathjax: true
tags: 
  - latex
  - math
has_children: true
---

# Limit Surface Planner

## Understanding Dual Limit Surfaces

Dual limit surfaces help encode the relationship between rotary motion and translational motion during planar sliding. 

A limit surface is described as:

$$
\begin{align}
\mathbf{w}^T \mathbf{A}+ \mathbf{w} = 1 \\
\mathbf{A} = \begin{bmatrix}
\frac{1}{(\mu F_N)^2} & 0 & 0 \\
0 & \frac{1}{(\mu F_N)^2} & 0 \\
0 & 0 & \frac{1}{(\mu r c F_N)^2}
\end{bmatrix}
\end{align}
$$

where $$\mathbf{w} = [F_x, F_y, \tau]^T$$ represents the generalized friction force (with $$F_x, F_y$$ being the frictional forces in the x and y directions, and $$\tau$$ being the torque about the vertical axis), and $$A$$ is a positive definite matrix that defines the shape of the limit surface. $$c \approx 0.6$$.

Using maximum dissipation principle, we can relate the generalized velocity $$\mathbf{v} = [V_x, V_y, \omega]^T$$ to the generalized friction force $$\mathbf{w}$$ as follows:

$$
\begin{align*}
\mathbf{w}^T \mathbf{A} \mathbf{w} &= 1 \\
\lambda \frac{\partial}{\partial \mathbf{w}} (\mathbf{w}^T \mathbf{A} \mathbf{w}) &= \mathbf{v} \\
\mathbf{A} = \mathbf{A}^T \space &\text{by definition.} \\
2 \lambda \mathbf{A} \mathbf{w} &= \mathbf{v} \\
\mathbf{w} &= \frac{1}{2 \lambda} \mathbf{A}^{-1} \mathbf{v} \\
\text{Substituting into } \mathbf{w}^T \mathbf{A} \mathbf{w} &= 1 \\
\left(\frac{1}{2 \lambda} \mathbf{A}^{-1} \mathbf{v}\right)^T \mathbf{A} \left(\frac{1}{2 \lambda} \mathbf{A}^{-1} \mathbf{v}\right) &= 1 \\
\frac{1}{4 \lambda^2} \mathbf{v}^T \mathbf{A}^{-1} \mathbf{v} &= 1 \\
\lambda &= \frac{1}{2} \sqrt{\mathbf{v}^T \mathbf{A}^{-1} \mathbf{v}} \\
\mathbf{w} &= \frac{\mathbf{A}^{-1} \mathbf{v}}{\sqrt{\mathbf{v}^T \mathbf{A}^{-1} \mathbf{v}}}
\end{align*}
$$

The final equation is now:
$$
\begin{equation}
\mathbf{w} = \frac{\mathbf{A}^{-1} \mathbf{v}}{\sqrt{\mathbf{v}^T \mathbf{A}^{-1} \mathbf{v}}}
\end{equation}
$$

For dual limit surfaces, we have two limit surfaces (two patch contacts). We use this to compute a velocity constraint for no-slippage planning.

$$
\begin{equation}
\mathbf{w_1}^T \mathbf{A}_1 \mathbf{w_1} = 1 \\
\mathbf{w_2}^T \mathbf{A}_2 \mathbf{w_2} < 1 \\
\text{Force balance gives us} \space \mathbf{w}_1 + \mathbf{w}_2 = 0
\end{equation}
$$

where you want to slip on $$A_1$$ and not slip on $$A_2$$. This gives you a velocity constraint if you plug in the relationship between $$\mathbf{w}$$ and $$\mathbf{v}$$.

## Bimanual Dual Limit Surfaces

So the difference here is our force balance equation will have gravity affect the object.

$$
\begin{align*}
\mathbf{w}_a \mathbf{A}_a \mathbf{w}_a &= 1 \\
\mathbf{w}_b \mathbf{B}_b \mathbf{w}_b &< 1 \\
\mathbf{w}_a + \mathbf{w}_b + \mathbf{g}_f = 0
\end{align*}
$$

where $$\mathbf{g}_f$$ is projection of gravity onto the plane of contact.

$$
\begin{align*}
\mathbf{w}_a \mathbf{B} \mathbf{w}_a + 2 \mathbf{g}_f^T \mathbf{B} \mathbf{w}_a + \mathbf{g}_f^T \mathbf{B} \mathbf{g}_f < 1 \\
\mathbf{w}_a^T (\mathbf{B} - \mathbf{A}) \mathbf{w}_a + 2 \mathbf{g}_f^T \mathbf{B} \mathbf{w}_a + \mathbf{g}_f^T \mathbf{B} \mathbf{g}_f < 0 \\
\mathbf{v}^T (\mathbf{A}^{-1} - \mathbf{B} \mathbf{A}^{-1} - \mathbf{A}^{-1}) \mathbf{v} - 2 (\sqrt{\mathbf{v}^T \mathbf{A}^{-1} \mathbf{v}}) \mathbf{v}^T \mathbf{A}^{-1} \mathbf{B} \mathbf{g}_f + (\mathbf{g}_f^T \mathbf{B} \mathbf{g}_f) (\mathbf{v}^T \mathbf{A}^{-1} \mathbf{v}) < 0
\end{align*}
$$


By considering special cases (like patch contact radius is the same, or friction coefficients are the same), we can simplify the above equation.

However, people have found this use case too limiting. This is where the project died.