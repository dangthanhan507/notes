---
layout: default
title: Isaac Lab
parent: Code Notes
nav_order: 20
has_children: true
---

# Isaac Lab

I am currentlyy migrating from Isaac Gym to Isaac Lab. These notes will document the process of using Isaac Lab.


## Isaac Ecosystem

**Isaac Gym** is GPU-based physics simulation for robot learning. It is really old and terrible. It has been deprecated since. This current setup does not support ROS, deformable object interactions, and high-fidelity rendering. 

**Isaac Sim** is the new shit. OmniIsaacGymEnvs is the replacement for IsaacGymEnvs when using Isaac Sim. 

**Isaac Lab** is the replacement for OmniIsaacGymEnvs and is the single robot learning framework for Isaac Sim. 