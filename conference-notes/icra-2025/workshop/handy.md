---
layout: default
title: Handy Workshop
parent: ICRA 2025
nav_order: 1
mathjax: true
tags: 
  - latex
  - math
---

# Handy Workshop (5/19/2025)

## Shuran Song

Research tries to tackle the cross-embodiment problem. 


<b> Cross-embodiment </b> 

2 ways they're approaching this:
1. Hardware design an interface to get around embodiment gap. (DexUMI)
2. Train a policy that accounts for this variation (GET-Zero).

Note for (2), this is still limited to a certain setup. Shuran wants to extend the capabilities of (2).


## NVIDIA: Ankur Handa

These guys I've seen in terms of Ratliff's work.

In summary, it is interesting to check out their work. 

Their papers:
1. Dextreme
2. DextremeFGP
3. DextrAH
4. DextrAH-RGB
5. DextrAH-MTS
6. DexPBT

Concepts that are re-occurring:
1. Automatic Domain Randomization
2. Teacher-Student

## Tengyu Liu

<b> NOTE: </b> Didn't pay enough attention.

Use Mocap with hand tracking to get reference traj. Use that to learn a policy?

Papers that they presented:
1. ManipTrans
2. DexManipNet

## Spotlight

### SeqGrasp: Sequential Multi-object GRASP

<b> Dataset Generation: </b> Initialize random robot hand poses around object. Run energy-based optimization to get a grasp.

Multi-object grasping is about grasping multiple objects in one go.

### Dexonomy

Another way to grasp with multi-finger hand. 

### NormalFlow

This is a tracking paper using visuotactile. They minimize Normal Map difference instead of doing point cloud matching (ICP). This is done with unconstrained optimization (Gauss-Newton Method).

### DOGlove

It's a teleop glove. 

Kili what're your thoughts?: Shit - Kili 2025
- Under 600

## Xiaolong Wang

Humanoid policy $$\sim$$ Human Policy. 

Does some collection with Egocentric data.
Sleeper guy.


