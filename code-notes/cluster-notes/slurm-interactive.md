---
layout: default
title: SLURM Interactive
parent: Cluster Notes
nav_order: 8
---

# Using SLURM Interactively

Something that would be very cool to achieve is using the cluster like a local machine.

One way of doing this is to use `salloc` to allocate resources and then interact with them on the terminal.

We could then run a setup script to load modules, setup the right files, and then run a command like `tmux` to keep the session alive.

Reminder on the role of directories:
- `/scratch` is for temporary files, intermediate data, IO operations for slurm job. It is purged
- `/home` is for user files or scripts. It is not purged.
- `/tmp` is local files that can be quickly accessed in the node. It is purged after the job is done or process dies.

## GUI Visualization

Visualizing the GUI of a program on a remote server is possible using X11 forwarding.
This is similar to specifying `-X` or `-Y` when using `ssh`. Then adding in the display variable to the command.

Alternatives include NX (which anydesk uses) or lsh.

**NOTE**: After much digging, it seems that visualizing GUI applications on the Great Lakes cluster is not recommended. We should treat Great Lakes like a prod cluster. Fundamentally, we have to pay for resources on the cluster, and using it for visualization (which is what a GUI does), is wasting resources that could be used for actual compute/training. Any additional notes here can be used for other local lab machines like exxact/lambda.

## Using anaconda/mamba/miniconda

You'll notice when using these that conda environments are saved in the `/home` directory. This will easily start cluttering your home directory.

We can think about storing anaconda environments in `/scratch` or `/tmp` instead.


Using `/tmp` would containerize the environment to the node, but it would be purged after the job is done. We would have to reinstall packages every time (maybe waste of time). One big benefit is that it is local to the node, so there is no network latency/overhead.
