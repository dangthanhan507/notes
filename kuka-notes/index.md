---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Kuka FRI Notes
math: katex
has_children: true
---

My notes on Kuka Iiwa robots, Drake, and FRI software.

FRI stands for Fast Robot Interface. It is a software interface on the Kuka that allows for real-time control of the robot. It is a low-level backend accessible via the Kuka Sunrise.

The Kuka FRI driver has different control modes that are accessible:
- **Position Control**: This is the internal PID that controls robot joint positions accurate up to 1mm of error.
- **Impedance Control**: This is a control mode that allows for robot to be controlled in a compliant manner, where the dynamics for each joint joint act like a spring-damper system. You can feed feedforward joint torques and commanded joint positions to the robot. 
- **Torque Control**: This is the raw torque control mode that allows for direct control of the robot's joint torques. It is the most low-level control mode and requires careful tuning to avoid damaging the robot. It already takes into account the gravity compensation of the Kuka Robot.
- **Cartesian Impedance**: This is a control mode that allows for control of the robot's end-effector in Cartesian space. It is similar to impedance control, but it allows for control of the end-effector position and orientation in Cartesian space and spring-damper is in Cartesian space. NO Implementation for it currently in Drake.