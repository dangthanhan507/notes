---
layout: default
title: Control Barrier Functions
parent: Control Theory Notes
mathjax: true
tags: 
  - latex
  - math
has_children: true
---

# Control Barrier Functions

This is a quick note on Control Barrier Functions (CBF) from devansh's notes [here](https://dev10110.github.io/tech-notes/research/cbfs-simple.html). Honestly, he does a much better job explaining this than I ever could, so please check out his notes.

However, I would like to recap those notes and tie them into how it could be used for robot manipulators. CBFs are a way to ensure safety in a control system by enforcing constraints on the system's state. Particularly, we want a system to lie within a safe set which is defined by an inequality constraint. We also assume our system is control affine, meaning the dynamics can be expressed as:

$$
\begin{align}
\dot{\mathbf{x}} = f(\mathbf{x}) + g(\mathbf{x})\mathbf{u} \\
h(\mathbf{x}) \geq 0
\end{align}
$$

where $$h(\mathbf{x})$$ is a continuously differentiable function whose inequality defines the safety set. 

In order to ensure the system enforces $$h(\mathbf{x}) \geq 0$$, we can enforce something like the following:

$$
\begin{align}
\frac{d}{dt} h(\mathbf{x}) \geq -\alpha h(\mathbf{x})
\end{align}
$$

where $$\alpha$$ is a constant that defines how quickly we want to enforce the constraint. At $$h(\mathbf{x}) = 0$$, we get $$\dot{h}(\mathbf{x}) \geq 0$$, which means the system cannot go into the unsafe region (derivative won't allow it to decrease). If $$h(\mathbf{x}) = 5$$ and $$\alpha = 1$$, then we have $$\dot{h}(\mathbf{x}) \geq -5$$, which means the system can decrease but not too quickly. So in some ways, we are tuning how aggressively we want to approach the boundary of the safe set. The lower the $$\alpha$$, the less aggressively we are allowed to approach the boundary. The higher the $$\alpha$$, the more aggressively we can approach the boundary.

Lets include the system dynamics into the derivative of $$h(\mathbf{x})$$:

$$
\begin{align}
\frac{d}{dt} h(\mathbf{x}) &\geq -\alpha h(\mathbf{x}) \\
\frac{\partial h}{\partial \mathbf{x}} \dot{\mathbf{x}} &\geq -\alpha h(\mathbf{x}) \\
\frac{\partial h}{\partial \mathbf{x}} (f(\mathbf{x}) + g(\mathbf{x})\mathbf{u}) &\geq -\alpha h(\mathbf{x}) \\
\frac{\partial h}{\partial \mathbf{x}}f(\mathbf{x}) + \frac{\partial h}{\partial \mathbf{x}}g(\mathbf{x})\mathbf{u} &\geq -\alpha h(\mathbf{x})
\end{align}
$$

Note that this is linear w.r.t. $$\mathbf{u}$$. 

Back in the days when Aaron Ames came up with CBFs, he presented this idea to Ford (or some car company). They liked the idea, but they already had their own control stack. They did not want to change their entire control system to accommodate CBFs. Aaron Ames instead proposed to run CBFs as a filter on top of their existing controller. This means we can take the output of their existing controller $$\mathbf{u}_{des}$$ and find the closest control input $$\mathbf{u}$$ that satisfies the CBF constraint. This can be formulated as a quadratic program (QP):

$$
\begin{align}
\mathbf{u}^* = \arg\min_{\mathbf{u} \in \mathbb{R}^m} & \frac{1}{2} ||\mathbf{u} - \mathbf{u}_{des}||^2 \\
\text{s.t.} & \frac{\partial h}{\partial \mathbf{x}}f(\mathbf{x}) + \frac{\partial h}{\partial \mathbf{x}}g(\mathbf{x})\mathbf{u} \geq -\alpha h(\mathbf{x})
\end{align}
$$

## Difference between CBF and heuristics

Normally, when we write our controllers, we do collision checking and stop the robot right before it collides with an obstacle. This is a heuristic that resembles bang-bang control. However, if the robot is approaching the obstacle at rapid speeds, it would be almost impossible to stop beforehand. For a practical person, CBFs can be seen as a way of designing collision avoidance by incorporating the dynamics of the system itself and looking at the velocity of the system. This would ensure the robot would slow down as it approaches the obstacle, and stop before it collides.

## Application to robot manipulators

In dev's post, we already see that this can be applied to a quadrotor system. However, I would like to see how this can be applied to robot manipulators. Before, we used to write motion planners on robot arms and use positional collision-checking to make sure we stayed in our safe set. However, we are now entering an era where we are testing our "controllers" called policies (Reinforcement Learning / Imitation Learning) on real robots. For a lab, we want the confidence to run our policies on real robots without worrying about breaking the robots. Additionally, we don't want to change the underlying controller (due to restrictions on robot arm controllers or even reproducing results from other labs). CBFs can be a quick way to add a safety filter on top of our existing controller (which could be a learned policy) to ensure safety.

Robot manipulators follow the following dynamics:
$$
\begin{align}
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \mathbf{u} \\
\dot{\mathbf{x}} = \begin{bmatrix} \dot{\mathbf{q}} \\ \ddot{\mathbf{q}} \end{bmatrix} = \begin{bmatrix} \dot{\mathbf{q}} \\ \mathbf{M}(\mathbf{q})^{-1}(-\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} - \mathbf{g}(\mathbf{q}) + \mathbf{u}) \end{bmatrix} \\
\dot{\mathbf{x}} = \begin{bmatrix} \dot{\mathbf{q}} \\ \ddot{\mathbf{q}} \end{bmatrix} = \begin{bmatrix} \dot{\mathbf{q}} \\ \mathbf{M}(\mathbf{q})^{-1}(-\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} - \mathbf{g}(\mathbf{q})) \end{bmatrix} + \begin{bmatrix} 0 \\ \mathbf{M}(\mathbf{q})^{-1} \end{bmatrix}\mathbf{u}
\end{align}
$$

We have shown that these dynamics are control affine.

However, in most manipulator systems, we use cartesian impedance or joint impedance controllers and not just directly applying torque. The dynamics then follow a different form:

$$
\begin{align}
\mathbf{K}\mathbf{e} + \mathbf{D}\dot{\mathbf{e}} + \mathbf{M}(\mathbf{q})\ddot{\mathbf{e}} = \mathbf{\tau}_{\text{ext}} \\
\mathbf{e} = \mathbf{q}_{\text{des}} - \mathbf{q} \\
\end{align}
$$

Normally, $$\dot{\mathbf{q}}_{\text{des}} = 0$$ and $$\ddot{\mathbf{q}}_{\text{des}} = 0$$. We then get:

$$
\begin{align}
\mathbf{K}(\mathbf{q}_{\text{des}} - \mathbf{q}) - \mathbf{D}\dot{\mathbf{q}} - \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} = \mathbf{\tau}_{\text{ext}}
\end{align}
$$

For most policy learning frameworks, our actions are normally taken as $$\Delta \mathbf{q}$$. This leads to the following dynamics:

$$
\begin{align}
\mathbf{K}(\mathbf{q}_{\text{des}} + \Delta \mathbf{q} - \mathbf{q}) - \mathbf{D}\dot{\mathbf{q}} - \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} = \mathbf{\tau}_{\text{ext}} \\
\mathbf{K}(\mathbf{q}_{\text{des}} + \mathbf{u} - \mathbf{q}) - \mathbf{D}\dot{\mathbf{q}} - \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} = \mathbf{\tau}_{\text{ext}} \\
\ddot{\mathbf{q}} = \mathbf{M}(\mathbf{q})^{-1}(-\mathbf{\tau}_{\text{ext}} - \mathbf{D}\dot{\mathbf{q}} + \mathbf{K}(\mathbf{q}_{\text{des}} - \mathbf{q})) + \mathbf{M}(\mathbf{q})^{-1}\mathbf{K}\mathbf{u} \\
\end{align}
$$

We see that this is still control-affine. We can still throw in CBFs into our framework. While $$\mathbf{q}_{\text{des}}$$ depends on what $$\mathbf{u}$$ is, we can treat $$\mathbf{q}_{\text{des}}$$ as a constant during the optimization since it is only a function of the previous state and action. Not only that but it is discretely updated, so until the next time we run the CBF-QP, it is effectively constant. 

## Flaws

We need to sysid the parameters of our robot. We also have to accrue our losses in $$\tau_\text{ext}$$ to account for any modeling errors from picking up objects or friction. Since we will effectively measure $$\tau_\text{ext}$$ as 0, this will affect the "stability" of our CBF. I don't think we can guarantee safety if our model is wrong. There are many ways to formulate CBFs to account for model uncertainty, but I won't dive into those for now.

## Operation Space Controllers (OSCs) + CBFs

We'll use OSC controllers for often than not. OSCs work in the end-effector space otherwise known as $$\text{SE}(3)$$. We can skip a few steps and show that the dynamics of the end-effector are:

$$
\begin{align}
\mathbf{\mathcal{F}} = \mathbf{K}_x(\mathbf{x}_d - \mathbf{x}) - \mathbf{D}_x\dot{\mathbf{x}} - \boldsymbol{\Lambda}(\mathbf{x})\ddot{\mathbf{x}} \\
\ddot{\mathbf{x}} = \boldsymbol{\Lambda}(\mathbf{x})^{-1}(-\mathbf{\mathcal{F}} - \mathbf{D}_x\dot{\mathbf{x}} + \mathbf{K}_x(\mathbf{x}_d - \mathbf{x})) \\
\ddot{\mathbf{x}} = \boldsymbol{\Lambda}(\mathbf{x})^{-1}(-\mathbf{\mathcal{F}} - \mathbf{D}_x\dot{\mathbf{x}} + \mathbf{K}_x(\mathbf{x}_d - \mathbf{x})) - \boldsymbol{\Lambda}(\mathbf{x})^{-1}\mathbf{K}_x\Delta \mathbf{x} 
\end{align}
$$

just like joint impedance, but now it's in end-effector space.

## Differential Inverse Kinematics + CBFs

One issue with OSCs is that we can blow fuses on our robot arms when we hit a singularity. This is because trying to achieve a certain position may require infinite joint effort which would blow the fuse (I've done it once). One way to get around this is to use differential inverse kinematics (DiffIK) wrapped around a position controller or joint impedance controller. The idea is that we want to achieve a delta end-effector pose $$\Delta \mathbf{x}$$ at every time step. This can be formulated as follows:

$$
\begin{align}
\dot{\mathbf{x}} = J(\mathbf{q})\dot{\mathbf{q}} \\
\Delta \mathbf{x} = J(\mathbf{q})\Delta \mathbf{q}
\end{align}
$$

we can use the pseudoinverse to find the appropriate $$\Delta \mathbf{q}$$ like so... $$\Delta \mathbf{q} = J(\mathbf{q})^{\dagger}\Delta \mathbf{p}$$ where $$\dagger$$ denotes pseudo-inverse.

Pseudo-inverses are the closed form solution to a least-squares problem. If we add constraints, we can get a QP for the DiffIK problem. But what is this actually? We are merely matching $$\Delta \mathbf{x}$$ as closely to $$\Delta \mathbf{x}_{\text{des}}$$ as possible. This means we can throw in CBF constraints into the QP to ensure safety. This is a really nice way to add safety without changing the underlying controller. Our QP would look something like this:

$$
\begin{align}
\Delta \mathbf{q}^* = \arg\min_{\Delta \mathbf{q} \in \mathbb{R}^n} & \frac{1}{2} ||J(\mathbf{q})\Delta \mathbf{q} - \Delta \mathbf{x}_{\text{des}}||^2 \\
\text{s.t.} & \frac{\partial h}{\partial \mathbf{x}}f(\mathbf{x}) + \frac{\partial h}{\partial \mathbf{x}}g(\mathbf{x})\Delta \mathbf{q} \geq -\alpha h(\mathbf{x}) \\
\mathbf{q}_{\text{min}} \leq & \mathbf{q} + \Delta \mathbf{q} \leq \mathbf{q}_{\text{max}} \\
\end{align}
$$

since we already know from joint impedance formulation that $$\Delta \mathbf{q}$$ is control affine.

Again, since our model is not perfect, we will not guarantee safety. However, this CBF is a first-order approach to safety (velocity) that can be nicely integrated into our existing controllers.

Compared to geometric fabrics which modifies the low-level controller, CBFs can be added on top of any existing controller. This is a really nice property that allows us to add "safety" to any existing controller without changing the underlying controller.

If we just stared at this optimization problem above without any priors to CBFs, we would even say it's a heuristical way of doing collision avoidance. 