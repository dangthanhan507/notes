---
layout: default
title: IsaacGymEnvs Setup
parent: Isaac
---

```bash
#!/bin/bash

cd ~
mkdir isaac
cd isaac
git clone git@github.com:MMintLab/TactileIsaacGymEnvs.git
git clone git@github.com:MMintLab/rl_games.git
# scp -r -P 2024 [uniqname]:141.212.84.140:/home/[uniqname]/[path_to_isaac]/IsaacGym_Preview_TacSL_Package/ ./
wget https://drive.google.com/file/d/12Sb5IwyP2YGtlmprybgdepWThHhsCmm7/view
tar -xf IsaacGym_Preview_TacSL_Package.tar.gz
rm IsaacGym_Preview_TacSL_Package.tar.gz

mamba create -n isaac python=3.8
source activate isaac

# setup IsaacGym
cd IsaacGym_Preview_TacSL_Package/isaacgym/python
python3 -m pip install -e .
cd ../../..

# setup rl_games
cd rl_games
python3 -m pip install -e .
cd ..

# setup IsaacGymEnvs
cd TactileIsaacGymEnvs
python3 -m pip install -e .
mkdir isaacgymenvs/runs
git checkout main

# install all the right packages
python3 -m pip install numpy==1.22.0
python3 -m pip install ray gymnasium scikit-image
python3 -m pip install torch_geometric
python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
python3 -m pip install moviepy rtree pytorch_volumetric urdfpy yourdfpy termcolor open3d opencv-python
```