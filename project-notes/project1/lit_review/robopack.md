---
layout: default
title: Robopack
parent: Lit Review (Graph Insertion)
nav_order: 1
---

# Robopack

Link: [Robopack](https://robo-pack.github.io)

Yunzhu's work on dense insertion. We learn the compliance of an object in the dense insertion task by taking multiple demos of us poking the bin objects. We then use this to learn the dynamics of the object and use MPC to insert the object into the bin by deforming the bin objects (pushing away other objects through extrinsic pushing). Extrinsic pushing is when we hold an object and that object pushes another object.

## Vague Ideas
- Use recurrent GNN to learn dynamics
- Use tactile information to help learn dynamics
- Use learned dynamics with sampling-based MPC to solve dense insertion

## Specifics
- **What does graph represent?**: the nodes of the graph are created by taking in multiple RGBD cameras in the scene and generating an object point cloud (through segmentation) and subsampling them. the edges of the graph are generated in the knn-graph scheme. We track the point clouds over time to ensure consistency foFor computational efficiency, we execute the first K planning steps. While executing the actions, the robot records its
tactile readings. After execution, it performs state estimation
with the history of observations and re-plans for the next
execution. More implementation details on planning can be
found in Appendix C.
To summarize this section, a diagram of the entire system
workflow including training and test-time deployment is available in Figure 10.r the recurrent GNN. We also include end-effector point clouds in the graph to show interaction.
- **What are tactile observations?**: We take the force distribution that can be estimated on the soft-bubble sensors and additionaly take point clouds from soft-bubble (depth sensor). this gives us a tactile signal that we can get embeddings from. 
- **How is compliance modelled?**: This is implicitly done through the recurrent GNN and observations. 

In the end, we're doing a fairly black-box dynamics learning model which uses GNNs. We also have a state-estimator that also uses GNNs. 

- **How can MPC figure out to make space?**: A special cost function is created that rewards exploring different starting actions. This is purely done heuristically. With different seeds, you can get different pokes with varying sucess.


## Limits

This takes FOREVER! I am pretty sure the dense packing was very hacky because they hard-coded the pokes to be between rows. Although it seems like a decent start since the MPC finds something. 


## Why does GNN work in this case?
- We get full object point clouds and end-effector point clouds that are all connected via knn-graph scheme. 
- We get point-wise tactile signals and map those into a latent embedding space. These embeddings are thought of as tactile observations which every node in the graph has as a part of its feature.

The compliance that we want for dense packing is learned through the data and observation (seeing tracked point clouds not behave like a rigid body). With enough pokes, we can also learn the tactile signals that are involved for poking and inserting successfully. 

## Questions related to my project
- Should we model multiple objects in this scene?
- How to model compliance explicitly?
- How to integrate extrinsic contact more explicitly and is that even useful?
- Why even use RL?