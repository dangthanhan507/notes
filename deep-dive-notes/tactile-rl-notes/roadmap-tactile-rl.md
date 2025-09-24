---
layout: default
title: Roadmap to Tactile RL
parent: Tactile RL Notes
mathjax: true
tags: 
  - latex
  - math
has_children: true
---

# Roadmap to Tactile

Now that we have some proper motivation on why to do reinforcement learning for manipulation, we should now motivate the use of tactile and its integration with RL.

We first define what tactile is. Tactile sensing is loosely spoken about the ability to sense contact and contact forces through a sensor. Normally, we can infer contact location, contact forces, and shear forces.

Being able to sense contact location and forces is important for manipulation tasks that involve a lot of forceful reasoning. Picking up objects and placing them is not really a forceful task. Tasks like inserting a key into a lock or screwing a bottle cap on a bottle are forceful tasks. While you could argue that you can do these tasks with vision, you won't have the same level of precision or robustness. When only doing vision, you really have to r ely on the precision of your robot/controller and have to work in high occlusion (e.g., hand occluding the object). Tactile can help with this by extending the sensing area to the fingers and providing direct contact information to something that is occluded.

There is a variety of tactile sensors being developed. There are currently force-torque sensors, tactile skin sensors (force arrays), and high-resolution visuotactile sensors (e.g., Gelsight, DIGIT, etc.). 

Force-torque sensors:
- Pros:
    - Very accurate and reliable force/torque measurements
- Cons:
    - Only measures at a single point.
    - Can't get contact location directly.

Visuotactile sensors:
- Pros:
    - We can use current vision tech to process tactile images to extract cues.
    - Very high dimension (definitely not losing out on information).
- Cons:
    - Sim2real is hard.
    - Requires much more processing power.

The others, I'm not too familiar with.

# Tactile BC

There have been works that use tactile sensing for behavior cloning. It is usually just another modality that is processed through MLP or CNN and then fused with other modalities at the embedding level. There have been successes, but there is not much literature on how well this can scale. Everyone has different sensors, so it is hard to collect huge amounts of data to try.

# Tactile RL

Before going into Tactile RL, these are the overarching challenges of RL for manipulation:
- Sample efficiency: RL is notoriously sample inefficient especially for robotics.
- Sim2real: How to transfer policies learned in simulation to the real world.
- State representation: How to represent the state of the world for manipulation tasks in simulation.

These are the general questions for Tactile Sensing (namely visuotactile):
- How to simulate visuotactile sensors?
- How to sim2real visuotactile sensors?

For our current interests, we want to sim2real tactile sensors.

Previously, we had discussed tactile resolution and how it is very high dimensional. One way of decreasing this is through the use of shear fields. The question then becomes how much resolution do we want.
- My question is how relevant this question is to answer... I don't think it's relevant at all.

