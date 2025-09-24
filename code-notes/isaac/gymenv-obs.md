---
layout: default
title: IsaacGymEnv Obs Notes
parent: Isaac
nav_order: 3
---

Now that we have a rudimentary understanding of the Isaac Gym API, we're going to dive into how `IsaacGymEnv` works. I'll do a backtracking approach. As in, I'll run code snippets and then figure out how each part works. The codebase is way too large, that I can only understand enough to get things working.

Current Issues with `TacSLIsaacGymEnv`:
- Wrist Wrench doesn't work (returns zeros)

# 1. Understanding TacSL Training Script

Currently, I can run the training script for TacSL peg insertion. My first goal is to try to run the training script and see if it works! Then I have to answer the following questions using the codebase:
- What is observation space? 
- What is the action space?
- What parts are being randomized between environments?
- What is the reward function?
- How does this connect with WandB?
- Where is policy gradient calculated?
- What is the policy architecture?
- How is GelSight included in the environment?

## 1.1 Training Script Arguments

These are the arguments:
- `checkpoint`: start training from specified checkpoint otherwise from scratch
- `task.env.numEnvs`: (int) number of environments to run in parallel
- `test`: (bool) whether to run eval or not
- `seed`: (int) random seed
- `max_iterations`: (int) number of training iterations
- `experiment`: (str) experiment name
- `task`: (str) task name to get environment from
- `task.env.task_type`: (str) task type to choose in environment (exclusive to tacsl)
- `task.rl.asymmetric_observations`: (bool) whether to use asymmetric observations
- `task.rl.add_contact_info_to_aac_states`: (bool) whether to add contact info to AAC states

## 1.2 Getting Observation

It seems that the observations are taken as a dictionary called `self.obs_dict`. 

We call observations in `post_physics_step` in `tacsl_task_insertion.py`:

```python
def post_physics_step(self):
    ...
    self.refresh_all_tensors()
    self.compute_observations()
    self.compute_reward()
    ...
```

Observations are taken by calling `compute_observations` in `tacsl_task_insertion.py`:

```python
    def compute_observations(self):
        """Compute observations."""

        if self.cfg_task.env.use_dict_obs:
            return self.compute_observations_dict_obs()

        return self.obs_buf  # shape = (num_envs, num_observations)
```

which calls `compute_observations_dict_obs`. This is the meat of the observation gathering code.



### 1.2.1 Tactile sensor force observation

You'll notice that we get contact observations implicitly through `plug_left_elastomer_force` which is the force sensor measured on the left elastomer on the jaw gripper. 

In `tacsl_task_insertion.py`, we see that:

```python
self.obs_dict['plug_left_elastomer_force'][:] = self.contact_force_pairwise[:, self.plug_body_id_env, self.franka_body_ids_env['panda_leftfinger']]
```
where `self.contact_force_pairwise` is assigned a value by the gym api:

```python
_contact_force_pairwise = self.gym.acquire_pairwise_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies * num_bodies, 3)
...
self.contact_force_pairwise = gymtorch.wrap_tensor(_contact_force_pairwise)
```
since this gym call wasn't reviewed in the tutorials we give a description from the documentation:

> Retrieves buffer for pairwise contact forces between all bodies in the sim. The buffer has shape (num_rigid_bodies, num_rigid_bodies, 3). Each contact force state contains one value for each X, Y, Z axis.

### 1.2.2 Pose observations

There are several poses observed in the observation space. These are a list of them:
- `plug`: thing to be inserted
- `ee`: end effector
- `socket`: the thing to be inserted into
- `dof_pos`: joint positions (revolute) + gripper position (prismatic)

Lets dive into how `ee_pos` is calculated. In `tacsl_task_insertion.py`, we see that:
```python
def compute_observations_dict_obs(self):
    self.obs_dict['ee_pos'][:] = self.fingertip_midpoint_pos
    ...
```

where `self.fingertip_midpoint_pos` is calculated in `tacsl_base.py`:

```python
def acquire_base_tensors(self):
    _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
    ...
    self.body_state = gymtorch.wrap_tensor(_body_state)
    ...
    self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
    ...
    self.fingertip_centered_pos = self.body_pos[:, self.fingertip_centered_body_id_env, 0:3]
    ...
    self.fingertip_midpoint_pos = self.fingertip_centered_pos.detach().clone()
    ...
```

And this is the gym documentaiton description of `self.gym.acquire_rigid_body_state_tensor`:

> Retrieves buffer for Rigid body states. The buffer has shape (num_rigid_bodies, 13). State for each rigid body contains position([0:3]), rotation([3:7]), linear velocity([7:10]), and angular velocity([10:13]).

This is the same way that `ee_quat` is calculated. Everything else is just indexing into the `self.body_state` tensor.

### 1.2.3 Joint observations

We get the `dof_pos` in `tacsl_task_insertion.py`:

```python
def compute_observations_dict_obs(self):
    ...
    if 'dof_pos' in self.cfg_task.env.obsDims or 'dof_pos' in self.cfg_task.env.stateDims:
        self.obs_dict['dof_pos'][:] = self.dof_pos
    ...
```

where `self.dof_pos` is calculated in `tacsl_base.py`:

```python
def acquire_base_tensors(self):
    ...
    _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)
    ...
    self.dof_state = gymtorch.wrap_tensor(_dof_state)
    ...
    self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
```

Pretty straightforward. This is all for the teacher training. We'll now move towards observations for the student training.

### 1.2.4 RGB observations 

These observations are listed as `left_tactile_camera_taxim` and `right_tactile_camera_taxim` in the observation space. These are the RGB images from the GelSight sensors with the shape (H,W,3) and are calculated in `tacsl_task_insertion.py`:

```python
def compute_observations_dict_obs(self):
    ...
    if self.cfg_task.env.use_camera_obs:
        images = self.get_camera_image_tensors_dict()
        if self.cfg_task.env.saveVideo and not self.cfg_task.env.use_tactile_field_obs:
            video_dict = {
                k: v for k, v in images.items()
                if k in [
                    'left_tactile_camera_taxim', 'right_tactile_camera_taxim',
                    'wrist', 'front', 'side'
                ]
            }
            frame = make_frame_from_obs(video_dict)
            self.frames.append(np.array(frame))
        
        ...

        if self.cfg_task.env.use_isaac_gym_tactile:
            # Optionally subsample tactile image
            ssr = self.cfg_task.env.tactile_subsample_ratio
            for k in self.tactile_ig_keys:
                images[k] = images[k][:, ::ssr, ::ssr]

        for cam in images:
            if cam in self.cfg_task.env.obsDims:
                if images[cam].dtype == torch.uint8:
                    self.obs_dict[cam][..., :3] = images[cam] / 255.
                else:
                    self.obs_dict[cam][..., :3] = images[cam]

        self.apply_image_augmentation_to_obs_dict()
```

So `self.obs_dict` takes in `images[cam]` which is filled in by `get_camera_image_tensors_dict` which is defined in `tacsl_base.py`:

```python
def get_camera_image_tensors_dict(self):
    """
    Get the dictionary of camera image tensors, including tactile RGB images.

    Returns:
        dict: Dictionary of camera image tensors.
    """
    camera_image_tensors_dict = super().get_camera_image_tensors_dict()

    # Compute tactile RGB from tactile depth
    if hasattr(self, 'has_tactile_rgb') and self.nominal_tactile:
        for k in self.nominal_tactile:
            depth_image = self.nominal_tactile[k] - camera_image_tensors_dict[k]    # depth_image_delta
            taxim_render_all = self.taxim_gelsight.render_tensorized(depth_image)
            camera_image_tensors_dict[f'{k}_taxim'] = taxim_render_all
    return camera_image_tensors_dict
```

where `camera_images_tensors_dict` is filled by `super().get_camera_image_tensors_dict()` which is defined in `tacsl_base.py`:

```python
def get_camera_image_tensors_dict(self):
    """
    Get the dictionary of camera image tensors.

    Returns:
        dict: Dictionary of camera image tensors.
    """
    # transforms and information must be communicated from the physics simulation into the graphics system
    if self.device != 'cpu':
        self.gym.fetch_results(self.sim, True)
    self.gym.step_graphics(self.sim)

    self.gym.render_all_camera_sensors(self.sim)
    self.gym.start_access_image_tensors(self.sim)

    camera_image_tensors_dict = dict()

    for name in self.camera_spec_dict:
        camera_spec = self.camera_spec_dict[name]
        if camera_spec['image_type'] == 'rgb':
            num_channels = 3
            camera_images = torch.zeros(
                (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1], num_channels),
                device=self.device, dtype=torch.uint8)
            for id in np.arange(self.num_envs):
                camera_images[id] = self.camera_tensors_list[id][name][:, :, :num_channels].clone()
        elif camera_spec['image_type'] == 'depth':
            num_channels = 1
            camera_images = torch.zeros(
                (self.num_envs, camera_spec.image_size[0], camera_spec.image_size[1]),
                device=self.device, dtype=torch.float)
            for id in np.arange(self.num_envs):
                # Note that isaac gym returns negative depth
                # See: https://carbon-gym.gitlab-master-pages.nvidia.com/carbgym/graphics.html?highlight=image_depth#camera-image-types
                camera_images[id] = self.camera_tensors_list[id][name][:, :].clone() * -1.
                camera_images[id][camera_images[id] == np.inf] = 0.0
        else:
            raise NotImplementedError(f'Image type {camera_spec["image_type"]} not supported!')
        camera_image_tensors_dict[name] = camera_images

    return camera_image_tensors_dict
```

where `camera_image_tensors_dict` is filled by `self.camera_tensors_list[id][name]` which monitors the images in the environment and fills in accordingly.

This process is set up in the following code in `tacsl_sensors.py`:

```python
class CameraSensor:
    """
    Class for managing camera sensors.
    Provides methods for creating and managing camera actors and tensors.
    """
    def create_camera_actors(self, camera_spec_dict):
        """
        Create camera actors based on the camera specification dictionary.
        # Note: This should be called once, as IsaacGym's global camera indexing expects all cameras of env 0 be created before env 1, and so on.

        Args:
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            tuple: List of camera handles and list of camera tensors.
        """
        camera_handles_list = []
        camera_tensors_list = []

        for i in range(self.num_envs):
            env_ptr = self.env_ptrs[i]
            env_camera_handles = self.setup_env_cameras(env_ptr, camera_spec_dict)
            camera_handles_list.append(env_camera_handles)

            env_camera_tensors = self.create_tensors_for_env_cameras(env_ptr, env_camera_handles, camera_spec_dict)
            camera_tensors_list.append(env_camera_tensors)
        return camera_handles_list, camera_tensors_list

    def create_tensors_for_env_cameras(self, env_ptr, env_camera_handles, camera_spec_dict):
        """
        Create tensors for environment cameras.

        Args:
            env_ptr: Pointer to the environment.
            env_camera_handles (dict): Dictionary of camera handles.
            camera_spec_dict (dict): Dictionary of camera specifications.

        Returns:
            dict: Dictionary of environment camera tensors.
        """
        env_camera_tensors = {}
        for name in env_camera_handles:
            camera_handle = env_camera_handles[name]
            if camera_spec_dict[name].image_type == 'rgb':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_COLOR)
            elif camera_spec_dict[name].image_type == 'depth':
                # obtain camera tensor
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                     gymapi.IMAGE_DEPTH)
            else:
                raise NotImplementedError(f'Camera type {camera_spec_dict[name].image_type} not supported')

            # wrap camera tensor in a pytorch tensor
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)

            # store references to the tensor that gets updated when render_all_camera_sensors
            env_camera_tensors[name] = torch_camera_tensor
        return env_camera_tensors
```

We can see that `env_camera_tensors` are filled automatically when `render_all_camera_sensors` is called.

<b> NOTE: </b> Now, we move onto how to get shear fields. This has already been implemented in the codebase, we are just finding where it is located and how to use it.

### 1.2.5 Shear field observations

We can find a way to include shear fields in the observation space by looking at `tacsl_task_insertion.py` in the `compute_observations_dict_obs` function:

```python
def compute_observations_dict_obs(self):
    ...
    if self.cfg_task.env.use_shear_force:
        tactile_force_field_dict = self.get_tactile_force_field_tensors_dict(debug=True if self.cfg_task.env.use_tactile_field_obs else False)
        # import ipdb; ipdb.set_trace()
        if self.cfg_task.env.use_tactile_field_obs:
            for k in ['tactile_force_field_left', 'tactile_force_field_right']:
                self.obs_dict[k][:] = tactile_force_field_dict[k]
                if self.cfg_task.env.zero_out_normal_force_field_obs:
                    self.obs_dict[k][..., 0] *= 0.0
        ...
```

The two arguments we want for observation space are `tactile_force_field_left` and `tactile_force_field_right`. Of course, all of these values are computed in `self.get_tactile_force_field_tensors_dict` which is defined in `tacsl_env_insertion.py`:

```python
def get_tactile_force_field_tensors_dict(self, debug=False):

    tactile_force_field_dict_raw = self.get_tactile_shear_force_fields()
    tactile_force_field_dict_processed = dict()
    ... #filling in tactile_force_field_dict_processed
    return tactile_force_field_dict_processed
```

This depends on `self.get_tactile_shear_force_fields()` which is defined in `tacsl_sensors.py`:

```python
    def get_tactile_shear_force_fields(self):
        tactile_force_field = dict()
        for key, config in self.tactile_shear_field_configs_dict.items():
            indenter_link_id = config['indenter_link_rb_id']
            elastomer_link_id = config['elastomer_link_rb_id']
            result = self.get_penalty_based_tactile_forces(indenter_link_id, elastomer_link_id)
            tactile_force_field[key] = result
        return tactile_force_field
```

This calculates the shear fields which are not by default calculated in isaacgym.

** TLDR **:
- Use `tactile_force_field_left` and `tactile_force_field_right` in the observation space to get shear fields.

