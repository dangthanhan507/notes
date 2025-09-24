---
layout: default
title: IsaacGymEnv Assets Notes
parent: Isaac
nav_order: 7
---

# 5. Understanding Assets in IsaacGymEnv

**REMEMBER:**
- We can only load URDF and MJCF files in IsaacGym.
- The way that `isaacgym` loads assets is through `gym.load_asset`. 

We load plug asset and socket asset in `TacSLEnvInsertion` which inherits from `TacSLBase`.

In `TacSLBase`, we load the franka asset and table asset.

## 5.1 Plug and Socket Assets

The plug and socket assets are loaded in `TacSLEnvInsertion._import_env_assets` function. This function takes in a desired subassembly we specified such as the following:
- `round_peg_hole_4mm`
- `round_peg_hole_8mm`
- `round_peg_hole_12mm`
and so on.

When the code is creating the environments for the first time, it can use the choices of subassemblies to randomize what assets an environment will have. These assets do not change during the course of the process lifetime.

**NOTE:** It seems that the `TacSLEnvInsertion` and `TacSLTaskInsertion` classes have fundamentally hard-coded the `plug/socket` asset paradigm into their design. This means we cannot support multi-object insertion feature into this without significant change to these classes. It's much better fundamentally to create our own environment and our own task based on that environment. 

These assets are known in `TacSLEnvInsertion`. The yaml file containing the urdf paths for these assets are hard-coded in `self.asset_info_insertion` to a particular path. `IsaacGymEnvs/assets/tacsl/yaml/industreal_asset_info_pegs.yaml` is the path. 