---
layout: default
title: Beyond Pick and Place Workshop
parent: ICRA 2025
nav_order: 1
mathjax: true
---

# Beyond Pick and Place Workshop

## Katerina Fragkiadaki - 3D Generative Manipulation and Object Dynamics

Autonomous vehicles technology to robot manipulation.
- Foundation Models
- 2D diffusion policy or $$\pi_0$$
- 3D diffuser Actor

TO do 3d diffusion, predict gripper position 

- 3d point tracks
- model-based control with generative models (guided diffusion)

## David Held - Spatially-aware Robot Learning

- generalize dexterous manipulation to unseen objects?
- contact point network creates critic map (Q-value) .... gets contact locaiton through softmax
- motion network for actions (hybrid discrete-continuous RL)
    - learning hybrid actor-critic maps
- Visual Manip w/ legs
- HacMan++
- ArticuBot
    - generate lots of demos in simulation
- Point Cloud Diffusion Transformer (delta location of points)
    - this is SE(3) equivariant
    - 3D diffusion model
- TAPIP3D
- Discrete Diffusion


This is very similar to what I thought of (actor-critic maps)

## Rachel Holladay - Modelling Mechanics for Dexterous Decision-Making

Works on TAMP (Task And Motion Planning)

I think this is the same stuff as before.

New work (beyond pick and place)
- guarded planning
- GUARDED stands for: guiding uncertainty accounting for risk and dynamics

This is just revamped safety-based robotics.

## Micheal Posa - Dexterity and generalization?

- knows that robots will be first deployed doing tasks w/ BC or RL.
- pitching his stuff as forward-thinking

Model-based hierarchies struggle a lot with robotics. Micheal Posa thinks contact rich dynamics are a big problem.

Goes into Vysics which reconstructs objects using vision and contact-rich physics. 
Reconstructs objects with partial view and contact.

Russ said before -> Goal: "LQR" but for multi-contact systems

Can't solve the contact-rich MPC (QCQP or MIP).

Using linearization and ADMM you can split this into 2 parts:
- QP dynamic constraints
- complementarity constraint projection (non-convex)
- it works ok but is inherently local

Now he wants to do sampling with CI-MPC
- bi-level hybrid reasoning

<b> Insight: </b> Find an idea that you have unique insight into doing. If no one else did it, it wouldn't be solved. Don't chase the newest thing. 

## Spotlight

List of papers that are interesitng:
- GET-ZERO: Graph Embodiment Transformer for Zero-shot Embodiment Generalization
- Metric Semantic Manipulation-enhanced Mapping
- AugInsert: Learning robust force-visual policy via data augmentation
- FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation
- Streaming Flow Policy: Simplifying diffusion/flow policies
- DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation
- SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies
- Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation

## Russ Tedrake - Multitask pretraining

We got ourselves an LBM? He wants to make it sound more like a biology talk.

diffusion policy (DP) as single-task.

LBMs are multi-task version of DP.
- language conditioned visuomotor policy
- VLA is one way to make an LBM. 

Shows evidence of a long-horizon task (no decomposition):
- fine-tune LBM on a single task
- caveat: teleoperators are really good
- takes one-day of data to make it work.

**My Question**: Is LBM trained on same environment as his evidence video?

**Does LBM pretraining help on new single tasks?**
- the first principle is that you must not fool yourself and you are the easiest person to fool. (Richard Feynman)
- Evidence that pretraining makes dexterous manipulation better?

**Hypothesis 1:** Multitask scaling laws. By training on many tasks, we can add new tasks more quickly.
- less data for the same performance.

**Hypothesis 2:** Improved Robustness. Due to more diverse training data; skill transfer. 
- More data is more robustness.

Diffusion policy is scaled up for multitask:
- Resnet 18 -> CLIP-ViT/B-16 for RGB
- new CLIP-ViT/B-32 for language encoding (mostly frozen)
- UNet -> DiT
- ~150M -> ~560M parameters

Why focus on this?
- sota for dexterous tasks
- simpler to study than vlas

**Evaluation: Real-world hardware testing**: 
- A/B testing -> study relative scores (do everything in one day)
- always blind, rnadomized trials
- eval GUI for repeatable trials
- rich reporting on "rubrics" and post-hoc video analysis

**Evaluation: Simulation-based testing**:
- small number of high-quality scenes
- greater than 10 tasks per scene
- tasks is not visually obvious

**Sim+Real cotraining**:
- we don't use sim for datagen
- we cotrain with sim teleop to use sim for eval

**Eval: Simulation-based testing**:
- repeatable (up to GPU determinism)
- sim rollouts from last week are still useful today
- many more rollouts -> better statistics
- Runs automatically in cloud at interval checkpoints
- strong correspondence

**Statistical testing**:
- assuming i.i.d. bernoulli samples with unknown probability of success p. compute interval of confidence within 95% of p.
- note: testing checkpoints, not architectures
- many sources of randomness in pipeline: objects, environments, initial conditions.
- diffusion gives a stochastic policy
- randomness in training (not included in analysis)
- high variability across tasks

**How many rollouts do we need?**
- with N rollouts we can calculate expected value and variance
- rollouts for sim + real 
- sufficient condition for intervals 
- can we do better? separate two policies.
- we do bayesian analysis (violin plots)

**Experiment Design**:
- We're in low data regime
    - ~200 demos
    - vs. 2000-5000 from gemini-robotics
- many tasks are intentionally difficult
    - severe distribution shifts
    - diverse initial conditions
- philosophy
    - at 0% or 100% success, we have no signal
    - in the middle, we can see the difference

CLIP >> ResNet; DiT >> UNet; Relative Actions

Many architecture changes end up mostly in the noise:
- it's really statistically in the noise for many papers choosing certain architectures.
- hard to differentiate small changes

Hypothesis 1 was fucked for some tasks 
- it's because of the language steering

Hypothesis 1 finetuning ot unseen tasks:
- LBM got the same performance at only 15% of the data.

Hypothesis 2 was pretty good
- huge difference between singletask and fine-tuned multi-task.

10 rollouts isn't enough to be rigorous (it's just a vibe check).