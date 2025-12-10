---
layout: default
parent: Review
title: Systems Theory
nav_order: 0
has_children: true
mathjax: true
tags: 
  - latex
  - math
---

# Dynamical Systems Theory

This is the systems theory side of control theory. Here, we focus on dynamical systems and analyzing their stability. We can also analyze the controllability, observability, accessibility, and reachability of a system.


## Ordinary Differential Equations (ODEs)


Say we start off with an ODE of the form:

$$
\begin{align}
\dot{x} = f(x, u)
\end{align}
$$

The problem of finding the flow of a vector field is called the initial value problem (IVP). We usually find this using numerical methods like Euler's method or Runge-Kutta methods.

The important things we have to consider is whether this system admits a solution given an initial condition. We have to also consider if the solution is unique. There are a few theorems that can help us with this.

For existence of a solution, we can use the Peano existence theorem:

**Peano Existence Theorem:** 

If $$f(x, u)$$ is continuous on a region $R$ containing the initial condition $$(x_0, u_0)$$, then there exists at least one solution to the IVP. This is a local result. That is, it only guarantees the existence of a solution in a small neighborhood around the initial condition $$(t_0)$$.

**Picard-Lindelöf Theorem (or the Existence and Uniqueness Theorem):**

If $$f(x,u)$$ is continuous in $$t$$ and Lipschitz continuous in $$x$$ then there exists a unique solution in the local neighborhood of the initial condition $$t_0$$.

The standard proof for this theorem relies on us using the Banach fixed-point theorem and showing that the integral equation:

$$
x(t) - x(t_0) = \int_{t_0}^t f(x(\tau), u(\tau)) \mathop{d\tau}
$$

will converge to a unique solution $$x(t)$$.

## Stability of Dynamical Systems

### Lyapunov Stability

Lyapunov stability is defined as follows:

A system is Lyapunov stable if for every $$\epsilon > 0$$, there exists a $$\delta > 0$$ such that if $$\|x(t_0)\| < \delta$$, then $$\|x(t)\| < \epsilon$$ for all $$t \geq t_0$$.

A system starting near the origin will remain near the origin. 

### Asymptotic Stability

A system is asymptotically stable if it is Lyapunov stable and if for every $$\epsilon > 0$$, there exists a $$\delta > 0$$ such that if $$\|x(t_0)\| < \delta$$, then $$\lim_{t \to \infty} \|x(t)\| = 0$$.

This is a stronger condition which entails that the system trajectory will converge to the origin.

### Exponential Stability

Same as asymptotic stability, but with an exponential rate of convergence. This means that the system will converge in finite time.


While we can analyze all trajectories of a system, it is often easier to analyze the stability of the system using Lyapunov functions. This is the Lyapunov direct method. We can use the Lyapunov direct method to analyze the stability of a system without having to solve for the trajectories of the system.

### Lyapunov Direct Method

The Lyapunov direct method involves finding a Lyapunov function $$V(x)$$ that satisfies the following conditions:
- $$V(x)$$ is positive definite.
- The derivative of $$V(x)$$ along the trajectories of the system, denoted as $$\dot{V}(x)$$, is negative semi-definite.

If these conditions are met, then the system is Lyapunov stable. If $$\dot{V}(x)$$ is negative definite, then the system is asymptotically stable. 

If $$\dot{V}(x) \leq -\alpha V(x)$$ for some positive constant $$\alpha$$, then the system is exponentially stable.

In the olden days, we would find a Lyapunov function by trial and error. We would also find a control law $$u(x)$$ using Control Lyapunov functions (Backstepping, CLF, Sliding Mode Control, etc.).

### LaSalle's Invariance Principle

Sometimes we will get a Lyapunov function that is not negative definite. In this case, we can use LaSalle's invariance principle to analyze the stability of the system.

LaSalle's invariance principle states that if $$\dot{V}(x) \leq 0$$ and the set of points where $$\dot{V}(x) = 0$$ is invariant under the flow of the system, then the system will converge to the largest invariant set within this set. If this set is a single point, then the system is asymptotically stable.

### Chetaev's Theorem

Chetaev's theorem is used to prove instability. If we can find a function $$V(x)$$ that is positive definite and whose derivative along the trajectories of the system is positive definite, then the system is unstable. This is the opposite definition of Lyapunov stability.

### Controllability

### Reachability

### Accessibility

## Special Case - Linear Time-Invariant (LTI) Systems

