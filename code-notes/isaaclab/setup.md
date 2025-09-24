---
layout: default
title: Isaac Lab Setup
parent: Isaac Lab
---

# Isaac Sim Install

## Pip Install

```bash
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
pip install isaaclab[isaacsim,all]==2.1.0 --extra-index-url https://pypi.nvidia.com
pip install git+https://github.com/isaac-sim/rl_games.git
pip install isaaclab==2.1.0 --extra-index-url https://pypi.nvidia.com
```

Test if it works with `isaacsim`.

Make sure torch-scatter matches pytorch version

get rsl-rl-lib and skrl
