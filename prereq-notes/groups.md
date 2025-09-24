---
layout: default
title:  "GDL 1: Groups"
date:   2025-04-18
categories: GDL
parent: Prerequisites
# math: katex
mathjax: true
tags: 
  - latex
  - math
---

# GDL 1: Groups, Representation, and Equivariant Maps

## Symmetry Groups 

Example: 2D Rotation Group $$\text{SO}(2)$$
- $$\{\frac{\pi}{4}, \frac{\pi}{2}, \frac{3\pi}{2}, \pi, ...\}$$
- these rotations can be composed
- there is identity rotation
- each rotation has an inverse
- the compositions are associative


<b> Definition B.1 </b> (Group) <i> A group is a tuple $$(G,\cdot)$$, consisting of a set $$G$$ and a binary operation $$\cdot : G \times G \to G, (g,h) \mapsto g \cdot h$$ which satisfies the following three group axioms: </i>

$$
\begin{align*}
\text{associativity:} \space & \forall g,h,k \in G, (g \cdot h) \cdot k = g \cdot (h \cdot k) \\
\text{identity:} \space & \exists e \in G, \forall g \in G, g \cdot e = g = e \cdot g \\
\text{inverse:} \space & \forall g \in G, \exists g^{-1} \in G, g \cdot g^{-1} = e = g^{-1} \cdot g
\end{align*}
$$

Examples:
- translation group: $$(\mathbb{R}^d, +)$$
- unitary group: $$U(1) := \{ e^{i\phi} \mid \phi \in [0, 2\pi) \} $$
- general linear group: $$GL(d) := \{ g \in \mathbb{R}^{d \times d} \mid det(g) \neq 0 \} $$
- trivial group: \{ e \}

Counterexamples:
- cex1: $$ \{e^{i\phi} \mid \phi \in [0, \pi) \} $$
    - closure: $$e^{i\phi_1}e^{i\phi_2} = e^{i(\phi_1 + \phi_2)}, \phi_1 = \phi_2 = \frac{\pi}{2}$$
- cex2: $$ \{ g \in \mathbb{R}^{d \times d} \} $$
    - inverse: not all matrices have inverses

Groups may come with additional structure:
- Topological group:
    - structure on G: topology
    - binary operation: continuous map
- Lie Group:
    - structure on G: smooth manifold
    - binary operation: smooth map
- Finite Group:
    - structure on G: finite set
    - binary operation: any function between finite sets

## Abelian Groups

<b> NOTE </b> we do not assume commutativity in the definition of a group.

<b> Definition B.2 </b> (Abelian Group) <i> A group $$(G,\cdot)$$ is called abelian if $$\forall g,h \in G, g \cdot h = h \cdot g$$ </i>


## Subgroups

<b> Definition B.3 </b> (Subgroup) <i> A subset $$H \subseteq G$$ is called a subgroup of $$G$$ if $$H$$ is closed under composition and inverses: </i>

- composition: $$\forall g,h \in H$$ one has $$gh \in H$$
- inverse: $$\forall g \in H$$ one has $$g^{-1} \in H$$

Subgroups are themselves also groups. Usual notation is $$H \leq G$$.

Examples:
- discrete translations: $$(\mathbb{Z}^d, +) \leq (\mathbb{R}^d, +)$$
- rotations: $$\text{SO}(2) \leq \text{SO}(3)$$
- trivial: $$\{e \} \leq G$$

## Product of Groups
<b> Definition B.4 </b> (Product of Groups) <i> Let $$(G_1, \cdot_1)$$ and $$(G_2, \cdot_2)$$ be two groups. The product group $$(G_1 \times G_2, \cdot)$$ is defined as follows: </i>

$$
\begin{equation*}
G_1 \times G_2 \to G_1 \times G_2, \quad ((g_1,g_2), (h_1,h_2)) \mapsto (g_1 \cdot_1 h_1, g_2 \cdot_2 h_2)
\end{equation*}
$$

Example:
- Cylinder Symmetry Group: $$\text{SO}(2) \times (\mathbb{R}, +)$$

Counterexample:
- 2D Rigid Transform: $$\text{SO}(2) \times (\mathbb{R}^2,+)$$
    - this is a semi-direct product, not a direct product

## Group Homomorphisms

<b> Definition B.5 </b> (Group Homomorphism) <i> A group homomorphism $$\phi: G_1 \to G_2$$ is a map between two groups $$(G_1, \cdot_1)$$ and $$(G_2, \cdot_2)$$ such that $$\forall g,h \in G_1$$ one has $$\phi(g \cdot_1 h) = \phi(g) \cdot_2 \phi(h)$$ </i>

This implies that $$\phi(g^{-1}) = \phi(g)^{-1}$$ and that $$g(e_{G_1}) = e_{G_2}$$.

Example:
- complex exponentiation: $$\text{exp}(i(\cdot)): \quad (\mathbb{R}, +) \mapsto U(1)$$, $$x \mapsto e^{ix} $$

We can lose information about a group. as seen with complex exponentiation. It wraps a real number along a circle.

## Group isomorphisms

<b> Definition B.6 </b> (Group Isomorphism) <i> A group isomorphism $$\phi: G_1 \to G_2$$ is a bijective group homomorphism. As in, it is an invertible group homomorphism. </i>

## Group Actions

<b> Definition B.7 </b> (Group Action) <i> A left group action of a group $$(G, \cdot)$$ on a set $$X$$ is a map $$\star: G \times X \to X$$ such that: </i>

- associativity: $$(g \cdot h) \star x = g \star (h \star x)$$ $$\forall g,h \in G$$, $$x \in X$$
- identity: $$e \star x = x$$, $$\forall x \in X$$

## Group orbit

<b> Definition B.8 </b> (Group Orbit) <i> The orbit of an element $$x \in X$$ under the left action $$\star$$ of $$G$$ is the set: </i>

$$
\begin{equation*}
G \star x = \{ g \star x \mid g \in G \}
\end{equation*}
$$

The orbit is different depending on which element we are operating on. For example, Let $$G = \text{SO}(2)$$ and $$X = \mathbb{R}^2$$, then the orbit of $$x = (1,0)$$ is the circle of radius 1. The orbit of $$x = (0,1)$$ is also the circle of radius 1. The orbit of $$x = (2,0)$$ is the circle of radius 2.


### Equivalence Relation

"being in the same orbit" defines an equivalence relation.
- relfexivity: $$x \sim_\star x$$ that is, x is contained in its own orbit $$G \star x$$.
- symmetry: $$x \sim_\star y \iff y \sim_\star x$$
- transitivity: $$x \sim_\star y \land y \sim_\star z \implies x \sim_\star z$$
    - if $$x$$ is in the same orbit as $$y$$ and $$y$$ is in the same orbit as $$z$$, then $$x$$ is in the same orbit as $$z$$.

These 3 properties define an equivalence relation.

## Quotient Set andd Quotient Map

<b> Definition B.9 </b> (Quotient Set) <i> The quotient set induced by G-action $$\star$$ on $$X$$ is the set of all orbits:

$$
\begin{equation*}
G \backslash X = \{ G \star x \mid x \in X \}
\end{equation*}
$$

The corresponding quotient map collapses elements of $$X$$ into their orbit:

$$
\begin{equation*}
q_\star: X \to G \backslash X, \quad x \mapsto G \star x
\end{equation*}
$$

## Transitive Action / Homogeneous Space

<b> Definition B.10 </b> (Transitive Action) <i> A group action $$\star$$ is called transitive if $$\forall x,y \in X$$ there exists $$g \in G$$ such that $$g \star x = y$$. </i>

$$X$$ is then called a homogeneous space.

## Stabilizer Subgroup

<b> Definition B.11 </b> (Stabilizer Subgroup) <i> Let $$\star$$ be a G-action on a set $$X$$. The stabilizer subgroup of $$x \in X$$ is defined as: </i>

$$
\begin{equation*}
\text{Stab}_x := \{ g \in G \mid g \star x = x \} \leq G
\end{equation*}
$$

## Invariant Map

<b> Definition B.12 </b> (Invariant Map) <i> A map $$f: X \to Y$$ is called invariant under the action $$\star$$ if $$\forall g \in G, x \in X$$ one has $$f(g \star x) = f(x)$$. </i>

Invariant maps "descent to the quotient". i.e. for any G-invariant map $$L: X \to Y$$, there exists a unique map $$\tilde{L}: G \backslash X \to Y$$ such that $$L = \tilde{L} \circ q_\star$$.

## Equivariant Maps

<b> Definition B.13 </b> (Equivariant Map) <i> A map $$f: X \to Y$$ is called equivariant under the action $$\star$$ if $$\forall g \in G, x \in X$$ one has $$f(g \star_X x) = g \star_Y f(x)$$. </i>

Most popular example is convolutions:

$$
\begin{align*}
K*: L^2(\mathbb{R}) \to L^2(\mathbb{R}), f \mapsto K * f := \int_\mathbb{R} K(x-y)f(y)dy
\end{align*}
$$

$$
\begin{align*}
(K * (g \star f))(x) &= \int_\mathbb{R}K(x-y)(g \star f)(y) dy \\
&= \int_\mathbb{R} K(x-y)f(y-g)dy \\
&= \int_\mathbb{R} K((x-g)-z)f(z)dz \\
&= (K * f)(x-g) \\
&= (g \star (K * f) )(x)
\end{align*}
$$

So now we know that convolutions are translation equivariant.


## Group Representation Theory

<b> Definition B.14 </b> (Linear group representation) <i> A linear group representation of a group $$G$$ on a real vector space $$\mathbb{R}^N$$ is a group homomorphism

$$
\rho: G \to GL(R^N)
$$

Easy to think as way of mapping from group element to a matrix.

More topics on this area:
- Tensor Representations
- Intertwiner
- Irreducible Representations
- Isomorphic Representations
- Schur's Lemma
- Complete reducibility of unitary representations
- Clebsch-Gordan Decomposition
- Peter-Weyl Theorem and Fourier Transforms