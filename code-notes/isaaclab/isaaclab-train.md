---
layout: default
title: IsaacLab Training Notes
parent: Isaac Lab
---

# IsaacLab Training Notes

## Command to train

```bash
$ python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Armlab-PegInsert-Direct-v0 \
    --enable_cameras env.enable_tactile_sensor=true env.enable_obs_camera=true env.read_tactile_sensor=true \
    --num_envs 16  \
    --video \
    --device=cuda:0  env.finger_type=gs_mini env.robot_usd_path=franka_gelsight_mini_assembled_z13_x10.usd env.task.hand_init_pos=[0.0,0.0,0.057]
```

## Arguments Handling

We split arguments into two groups: arguments for `Applauncher` and arguments for Hydra configuration. We use `parse_known_args()` to separate them. The known arguments are passed to `AppLauncher`, while the remaining arguments are left for Hydra to process.

```python
import argparse
import sys
from distutils.util import strtobool
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument( ... )
parser.add_argument( ... )
parser.add_argument( ... )
...
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args() # AN NOTE: args_cli holds the known args, hydra_args holds the rest

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli) # pass args_cli to AppLauncher
simulation_app = app_launcher

...

def main(...):
    ...

if __name__ == '__main__':
    main()
    simulation_app.close()

```

## Hydra Main

