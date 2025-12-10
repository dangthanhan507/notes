---
layout: default
title: IsaacLab Main Script Notes
parent: Isaac Lab
---

# IsaacLab Main Script Notes

## command

```bash
python run_sample_isaaclab.py --task=Isaac-Armlab-PegInsert-Direct-v0 --num_envs 16
```

## run_sample_isaaclab.py
```python
import argparse
from isaaclab.app import AppLauncher
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser) # add extra arguments into parser

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg) # simulation starts here
    env.reset()
    i = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = 0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, reward, terminated, info = env.step(actions) # simulation is occurring here
            i += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
```

Since we supply `Isaac-Armlab-PegInsert-Direct-v0`, this is from `FactoryTaskPegInsertCfg`


We have three things that are going on here:
- `FactoryEnv(DirectRLEnv)` (env)
- `PegInsert` (task)

```python

class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        pass
    def _set_default_dynamics_parameters(self):
        pass
    def _init_tensors(self):
        pass
    def _setup_scene(self):
        pass
    def _compute_intermediate_values(self, dt):
        pass
    def _update_obs_history(self, obs_dict):
        pass
    def _get_factory_obs_state_dict(self):
        pass
    def _get_observations(self):
        pass
    def _reset_buffers(self, env_ids):
        pass
    def _pre_physics_step(self, action):
        pass
    def close_gripper_in_place(self):
        pass
    def _apply_action(self):
        pass
    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        pass
    def _get_dones(self):
        pass
    def _get_curr_successes(self, success_threshold, check_rot=False):
        pass
    def _log_factory_metrics(self, rew_dict, curr_successes):
        pass
    def _get_rewards(self):
        pass
    def _get_factory_rew_dict(self, curr_successes):
        pass
    def _reset_idx(self, env_ids):
        pass
    def _set_assets_to_default_pose(self, env_ids):
        pass
    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        pass
    def get_handheld_asset_relative_pose(self):
        pass
    def _set_franka_to_default_pose(self, joints, env_ids):
        pass
    def step_sim_no_action(self):
        pass
    def randomize_initial_state(self, env_ids):
        pass
```

This has all of the stuff the task code had in IsaacGym. This defines all of the functions that `DirectRLEnv` will use to implement the RL environment.

