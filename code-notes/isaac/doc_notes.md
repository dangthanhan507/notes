---
layout: default
title: Isaac Gym Doc Notes
parent: Isaac
nav_order: 2
---

# Isaac Gym Notes (for An)
 
This is the breakdown of the Isaac Gym codebase. Here are important links:
- [Documentation](https://docs.robotsfan.com/isaacgym/)
- [IsaacGym Forum](https://forums.developer.nvidia.com/c/robotics-edge-computing/isaac/isaac-gym/322)

---

## Vectorized Environments

IsaacGym environments are vectorized environments. This means they can run multiple environments in parallel. For us, this means that it follows the OpenAI Gym API for vectorized environments:


Here's example code from IsaacGymEnvs:

```python
import isaacgym
import isaacgymenvs
import torch

num_envs = 2000

envs = isaacgymenvs.make(
	seed=0, 
	task="Ant", 
	num_envs=num_envs, 
	sim_device="cuda:0",
	rl_device="cuda:0",
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(20):
	random_actions = 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
	envs.step(random_actions)
```

Here's another example from the Gym API site:

```python
envs = gym.vector.make("CartPole-v1", num_envs=3)
envs.reset()
actions = np.array([1, 0, 1])
observations, rewards, dones, infos = envs.step(actions)
```

Here's an example implementation of a Gym Environment:

```python
class DictEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "position": gym.spaces.Box(-1., 1., (3,), np.float32),
        "velocity": gym.spaces.Box(-1., 1., (2,), np.float32)
    })
    action_space = gym.spaces.Dict({
        "fire": gym.spaces.Discrete(2),
        "jump": gym.spaces.Discrete(2),
        "acceleration": gym.spaces.Box(-1., 1., (2,), np.float32)
    })

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        observation = self.observation_space.sample()
        return (observation, 0., False, {})
```

no extra methods are needed. We mainly focus on `reset` and `step` and make sure we define `observation_space` and `action_space` correctly. 

---

## 1. Simulation Setup

### 1.1 Gym
CoreAPI is located in `isaacgym.gymapi'. All Gym API functions are accessed by `Gym` object acquired by the following code:

```python
from isaacgym import gymapi
gym = gymapi.acquire_gym()
```

### 1.2 Sim

```python
sim_params = gymapi.SimParams()
sim = gym.create_sim(
    compute_device_id,
    graphics_device_id,
    gymapi.SIM_PHYSX,
    sim_params
)
```

we use `create_sim` to create a simulation which contains graphics context and physics context. Allows us to load assets, create environments, and interact with simulation.

We have two choices for simulation:
- `gymapi.SIM_PHYSX` - PhysX simulation: robust rigid body and articulation (CPU/GPU) with new tensor api support.
- `gymapi.SIM_FLEX` - Flex simulation: soft body and rigid body simulation (GPU only) does not fully support new tensor api.

### 1.3 Simulation Parameters

This is how to configure the simulation parameters:

```python
# get default set of parameters
sim_params = gymapi.SimParams()

# set common params
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX specific params
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex specific params
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.num_position_iterations = 4
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5

sim = gym.create_sim(
    compute_device_id,
    graphics_device_id,
    gymapi.SIM_PHYSX,
    sim_params
)
```

### 1.4 Ground Plane

```python
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) # z-up
plane_params.distance = 0.0
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
plane_params.restitution = 0.0

gym.add_ground(sim, plane_params)
```

### 1.5 Load Assets

Gym supports URDF and MJCF file formats. Here's a quick way to load those files into a simulation:

```python
asset_root = "../../assets"
asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)
```

The `load_asset` function uses the file name extension to determine which processing code to use. We can also pass extra information to the asset importer:

```python
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
```

This doesn't automatically add it to the simulation, but it is the first step.

### 1.6 Environment and Actors

Environment consists of collection of actors and sensors simulated together. The overall reason to pack them together is performance. 

Lets setup a simple environment:

```python
spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 8)
```
this creates 8 environments.

If we want to create an actor, we can do so below:
```python
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
actor_handle = gym.create_actor(
    env,
    asset,
    pose,
    "MyActor",
    0,
    1
)
```

Here's a quick example of going from axis-angle to quaternion:

```python
pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), -0.5 * math.pi)
```

Any argument after `MyActor` are optional. The two args are <b> collision_group </b> and <b> collision_filter </b>. The `collision_group` assigns the actor to a collision group. This prevents actors from the same environment from colliding with each other. The `collision_filter` is a bit mask that helps filter collision between bodies (kinda like preventing self-collision checks if we just have a robot). 

```python
num_envs = 64
envs_per_row = 8 #infer columns from num_envs
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []
actor_handles = []
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)
    
    height = random.uniform(1.0, 2.5)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 0.0)

    actor_handle = gym.create_actor(
        env,
        asset,
        pose,
        "MyActor",
        i,
        1
    )
    actor_handles.append(actor_handle)
```

Be aware that once you finish populating one environment and start creating another, the first environment can no longer have stuff added to it. This is because of how memory is allocated. We can find ways to "fake" having dynamic number of actors, but that is for later.

### 1.7 Running Sim

Super Simple:
```python
while True:
    # step the physics
    gym.simulate(sim) # each iteration is one timestep
    gym.fetch_results(sim, True)
```

### 1.8 Adding a Viewer

If we want to visualize our setup, we need to add a viewer. 

```python
cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
```

If we want to update the viewer (re-draw on it), we need to execute the following every iteration:

```python
gym.step_graphics(sim)
gym.draw_viewer(viewer, sim, True)
```

The `step_graphics` function synchronizes visual representation with simulation context. The `draw_viewer` renders the visual representation of the simulation. They are separate because `step_graphics` can be called without `draw_viewer` or even the need for a viewer.

To synchronize visual update frequency with real time we add this to the loop:

```python
gym.sync_frame_time(sim)
```

This throttles down the simulation rate to match real-time. If it's slower than real time, this has no effect.

We can terminate simulation by checking if viewer was closed with `query_viewer_has_closed`.

```python
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
```

---

## 2. Assets

Lets start off with loading an asset.

```python
asset_root = "../../assets"
asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset = gym.load_asset(sim, asset_root, asset_file)
```

We can specify options for asset.
```python
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01

asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
# asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE
# asset_options.convex_decomposition_from_submeshes = True
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
```

### 2.1 Overriding Inertial Properties

The URDF and MJCF file formats also specify inertial properties. We can override these properties.

```python
asset_options.override_com = True
asset_options.override_inertia = True
```

This will use values computed from geometries of collision shapes.

### 2.2 Convex Decomposition

```python
asset_options.vhacd_enabled = True
```

This enables convex decomposition which makes much more complex collision shape (matches actual mesh).

### 2.3 Procedural Assets (Primitives)

We can create simple geometric assets. Just need to do the following:

```python
asset_options = gym.AssetOptions()
asset_options.density = 10.0

box_asset     = gym.create_box(sim, width, height, depth, asset_options)
sphere_asset  = gym.create_sphere(sim, radius, asset_options)
capsule_asset = gym.create_capsule(sim, radius, length, asset_options)
```

---

## 3. Physics Simulation

### 3.1 Creating Actors

An actor is an instance of `GymAsset`. `create_actor` creates actor and adds it to environment and returns handle to it. Much better to save these handles rather than look them up every time we need to access them.

```python
envs = []
actor_handles = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(evs)

    # add actor
    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)
```

actor_handles are specific to an environment. Functions that operate on actor are `get_actor_*`, `set_actor_*`, and `apply_actor_*`.

### 3.2 Aggregates (only on PhysX)

Aggregate is collection of actors. They do not provide extra simulation functionality, but they tell you that a set of actors will be clustered together, which allow PhysX to optimize its spatial data operations. Creating them gives modest performance boost.

```python
gym.begin_aggregate(env, max_bodies, max_shapes, True)
gym.create_actor(env, ...)
gym.create_actor(env, ...)
gym.end_aggregate(env)
```

### 3.3 Actor Components

```python
num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
num_joints = gym.get_actor_joint_count(env, actor_handle)
num_dofs   = gym.get_actor_dof_count(env, actor_handle)
```

<b> Joints </b> - these can be fixed, revolute, prismatic.

<b> Degrees of Freedom (DOF) </b> - these are the actuated joints. They are the joints that can be controlled.


### 3.4 Controlling Actors

Controlling actors is done through using the DOFs. For each DOF, we can set drive mode, limits, stiffness, damping, and targets. 

<b> Scaling Actors </b> - we can scale their size at runtime. It changes its mas, joint positions, prismatic joint limits, and collision geometry. The CoM isn't updated though. Resetting transforms or velocities or applying forces at the same time as scaling may yield incorrect results. 

<b> DOF Properties and drive Modes </b> - can be accessed with `get_asset_dof_properties` and individual actors (`get_actor_dof_properties`/`set_actor_dof_properties`).

| Name | Data Type | Description |
|-------- | --------- | ----------- |
| hasLimits | bool | Whether DOF has limits |
| lower | float32 | Lower limit |
| upper | float32 | Upper limit |
| driveMode | gymapi.DofDriveMode | DOF drive mode |
| stiffness | float32 | Drive Stiffness |
| damping | float32 | Drive Damping |
| velocity | float32 | Max Velocity |
| effort | float32 | Max effort (force/torque) |
| friction | float32 | DOF friction |
| armature | float32 | DOF armature |

`DOF_MODE_NONE` lets joints move freely.

```python
props = gym.get_actor_dof_properties(env, actor_handle)
props['driveMode'].fill(gymapi.DOF_MODE_NONE)
props['stiffness'].fill(0.0)
props['damping'].fill(0.0)
gym.set_actor_dof_properties(env, actor_handle, props)
```

`DOF_MODE_EFFORT` lets us apply efforts to the DOF using `apply_actor_dof_efforts`. 

```python
# configure the joints for effort control mode (once)
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
props["stiffness"].fill(0.0)
props["damping"].fill(0.0)
gym.set_actor_dof_properties(env, actor_handle, props)

# apply efforts (every frame)
efforts = np.full(num_dofs, 100.0).astype(np.float32)
gym.apply_actor_dof_efforts(env, actor_handle, efforts)
```

`DOF_MODE_POS` lets u set target positions for a PD controller. 

```python
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(1000.0)
props["damping"].fill(200.0)
gym.set_actor_dof_properties(env, actor_handle, props)

targets = np.zeros(num_dofs).astype('f')
gym.set_actor_dof_position_targets(env, actor_handle, targets)
```

if DOF is linear, target is in meters. If DOF is angular, target is in radians.

if we want to set random positions for the DOF, here it is:

```python
dof_props = gym.get_actor_dof_properties(envs, actor_handles)
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
ranges = upper_limits - lower_limits

pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype('f')
gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)
```

`DOF_MODE_VEL` is for velocity control.

```python
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props["damping"].fill(600.0)
gym.set_actor_dof_properties(env, actor_handle, props)

vel_targets = np.random.uniform(-math.pi, math.pi, num_dofs).astype('f')
gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)
```

Unlike efforts, the ones with controllers only require targets to be set every tame targets need change. Efforts need to be set every frame.

<b> NOTE: </b> Tensor Control API allows alternative ways to applying controls without setting in CPU. We can run simulations entirely on GPU.

### 3.5 Physics State

#### 3.5.1 Rigid Body States

```python
actor_body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
env_body_states = gym.get_env_rigid_body_states(env, gymapi.STATE_ALL)
sim_body_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL)
```

You can get states for an actor, an environment, or entire simulation.

This returns structured numpy arrays. Last argument is a flag for what it should return. `STATE_POS` only returns positions. `STATE_VEL` only returns velocities. `STATE_ALL` returns both positions and velocities.


We can access the states like so:

```python
body_states["pose"]             # all poses (position and orientation)
body_states["pose"]["p"])           # all positions (Vec3: x, y, z)
body_states["pose"]["r"])           # all orientations (Quat: x, y, z, w)
body_states["vel"]              # all velocities (linear and angular)
body_states["vel"]["linear"]    # all linear velocities (Vec3: x, y, z)
body_states["vel"]["angular"]   # all angular velocities (Vec3: x, y, z)
```

We can set the states like so:

```python
gym.set_actor_rigid_body_states(env, actor_handle, body_states, gymapi.STATE_ALL)
gym.set_env_rigid_body_states(env, body_states, gymapi.STATE_ALL)
gym.set_sim_rigid_body_states(sim, body_states, gymapi.STATE_ALL)
```

We can only find the index of a specific rigid body using the following:

```python
i1 = gym.find_actor_rigid_body_index(env, actor_handle, "body_name", gymapi.DOMAIN_ACTOR)
i2 = gym.find_actor_rigid_body_index(env, actor_handle, "body_name", gymapi.DOMAIN_ENV)
i3 = gym.find_actor_rigid_body_index(env, actor_handle, "body_name", gymapi.DOMAIN_SIM)
```

### 3.5.2 DOF States

We can also work with reduced coordinates. 

```python
dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)
```


```python
dof_states["pos"]   # all positions
dof_states["vel"]   # all velocities
```

---
## 4. Tensor API

This API use GPU-compatible data representations for interacting with simulations.

Tensors from this API are "global" tensors (they hold values for all actors in simulation). This sort of structure makes it very easy to enable data parallelism.

### 4.1 Simulation Setup

This is currently only available for PhysX simulation. 

```python
sim_params = gymapi.SimParams()
...
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True

sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
```

We must call `prepare_sim` to initialize internal data structures used by tensor API.

```python
gym.prepare_sim(sim)
```

### 4.2 Physics State

After `prepare_sim`, we can acquire physics state tensors. 

### 4.2.1 Actor Root State Tensor

A Gym actor has 1+ rigid bodies. All actors have a root body. The root state tensor holds state of all actor root bodies in simulation. The state of root body is 13-dimensional [position, quaternion, linear velocity, angular velocity]. 

```python
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
```

This returns generic tensor descriptor. In order to access contents of tensor, we warp it in PyTorch Tensor object using `gymtorch.wrap_tensor`.

```python
root_tensor = gymtorch.wrap_tensor(_root_tensor)
```

now this is a tensor in pytorch. here we can see the structure of the tensor:

```python
root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]
```

To update contents of tensors with latest state, call:

```python
gym.refresh_actor_root_state_tensor(sim)
```

this will update the tensor with latest state. Generally, do this after each call to `gym.simulate'.

```python
# ...create sim, envs, and actors here...

gym.prepare_sim()

# acquire root state tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)

# wrap it in a PyTorch Tensor and create convenient views
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]

# main simulation loop
while True:
    # step the physics simulation
    gym.simulate(sim)

    # refresh the state tensors
    gym.refresh_actor_root_state_tensor(sim)

    # ...use the latest state tensors here...
```

We can also set the root state tensor. 

```python
# acquire root state tensor descriptor
_root_tensor = gym.acquire_actor_root_state_tensor(sim)

# wrap it in a PyTorch Tensor
root_tensor = gymtorch.wrap_tensor(_root_tensor)

# save a copy of the original root states
saved_root_tensor = root_tensor.clone()

step = 0

# main simulation loop
while True:
    # step the physics simulation
    gym.simulate(sim)

    step += 1

    if step % 100 == 0:
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(saved_root_tensor))
```

We can also set root state tensor for a subset of actors using `set_actor_root_state_tensor_indexed`. 


### 4.2.3 Degrees of Freedom

Articulated actors have a number of DoFs whose state can be queried and changed. State of each DoF is represented with 2 float32 values: position and velocity. 

The DoF state tensor contains state of all DoFs in simulation. Number of DoFs can be obtained by calling `gym.get_sim_dof_count(sim)`. The DoF states are laid out sequentially. Number of DoFs for a specific actor can be obtained by calling `gym.get_actor_dof_count(env, actor)`. Global index of a specific DoF can be obtained a number of ways:
- `gym.get_actor_dof_index(env, actor_handle, i, gymapi.DOMAIN_SIM)`.
- `gym.find_actor_dof_index(env, actor_handle, dof_name, gymapi.DOMAIN_SIM)`

We can access dof state as follows:

```python
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
```

Function `refresh_dof_state_tensor` populates tensor with latest data from simulation.

just like the root state tensor, we can set the DoF state tensor using `gym.set_dof_state_tensor(sim, _dof_states)`. 

We can also set the DoF state tensor for a subset of actors using `set_actor_dof_state_tensor_indexed`.


### 4.2.4 Rigid Body States

Rigid body states contains state of all rigid bodies in simulation. Each rigid body has 13-dimensional state vector [position, quaternion, linear velocity, angular velocity]. 

We can get total number of rigid bodies using `gym.get_sim_rigid_body_count(sim)`.

We can get rigid body states from an actor using `gym.get_actor_rigid_body_states`.

We can get actor index using `gym.get_actor_rigid_body_index` or `gym.find_actor_rigid_body_index`.

We can get rigid body states using `gym.acquire_rigid_body_state_tensor(sim)`. Which will need to by wrapped using `gymtorch.wrap_tensor`.

```python
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)
```

We can populate tensor using `gym.refresh_rigid_body_state_tensor(sim)`.

### 4.2.5 Jacobian and Mass Matrices

In order to get these, you need the name of the actor which is provided when calling `create_actor`.

Once you have the name, you can use the following to get the Jacobian and Mass Matrix:

```python
_jacobian   = gym.acquire_jacobian_tensor(sim, "franka")
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")

# wrap tensor 
jacobian    = gymtorch.wrap_tensor(_jacobian)
mass_matrix = gymtorch.wrap_tensor(_massmatrix)
```

to refresh we call the following:

```python
gym.refresh_jacobian_tensors(sim, "franka")
gym.refresh_mass_matrix_tensors(sim, "franka")
```

Shape of Mass Matrix is (num_dofs, num_dofs). Shape of Mass Matrix Tensor is (num_envs, num_dofs, num_dofs).

Shape of jacobian depends on number of links, fixed-base or floating-base, and DOFs. Each body has 6 rows in jacobian representing linear and angular motions. The column represents the DOFs.

If actor base is floating, Jacobian shape is (num_envs, num_links, 6, num_dofs + 6). The first 6 columns are the base DOFs.

Here's an example of looking up jacobian of a link:

```python
link_dict = gym.get_asset_rigid_body_dict(franka_asset)
dof_dict = gym.get_asset_dof_dict(franka_asset)

link_index = link_dict["panda_hand"]
dof_index = dof_dict["panda_joint5"]

# for all envs:
jacobian[:, link_index, :, dof_index + 6]

# for a specific env:
jacobian[env_index, link_index, :, dof_index + 6]
```

If base is fixed, we don't have those extra 6 columns. Thus, the shape is (num_envs, num_links - 1, 6, num_dofs).

```python
# for all envs:
jacobian[:, link_index - 1, :, dof_index]

# for a specific env:
jacobian[env_index, link_index - 1, :, dof_index]
```

### 4.3 Contact Tensors

We can call the net contact forces experienced by each rigid body during last sim step. This is not each individual contact force, but the net force applied to the rigid body. 

```python
_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

# netcf shape: (num_rigid_bodies, 3)

gym.refresh_net_contact_force_tensor(sim)
```

### 4.4 Control Tensors

We show how to apply actions to simulation using tensor API.

```python
# get total number of DOFs
num_dofs = gym.get_sim_dof_count(sim)

# generate a PyTorch tensor with a random force for each DOF
actions = 1.0 - 2.0 * torch.rand(num_dofs, dtype=torch.float32, device="cuda:0")

# apply the forces
gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(actions))
```

Actuation forces will only be applied for DOFs whose driveMode was set to `DOF_MODE_EFFORT`.

We can also run `set_dof_position_target_tensor` or `set_dof_velocity_target_tensor` depending on our drive mode.

We can apply forces to rigid bodies using `apply_rigid_body_force_tensors`.

```python
# apply both forces and torques
gym.apply_rigid_body_force_tensors(sim, force_tensor, torque_tensor, gymapi.ENV_SPACE)

# apply only forces
gym.apply_rigid_body_force_tensors(sim, force_tensor, None, gymapi.ENV_SPACE)

# apply only torques
gym.apply_rigid_body_force_tensors(sim, None, torque_tensor, gymapi.ENV_SPACE)
```


### 4.5 Common Problem (Tensor Lifetime)

Tensor in PyTorch are subject to garbage collection. Gym C++ runtime is not conected with python interpreter, so it doesn't participate in the reference counting of objects. We should be careful to make sure objects are not garbage-collected.

```python
indices_torch = torch.LongTensor([0, 17, 42])
indices_gym = gymtorch.unwrap_tensor(indices_torch.to(torch.int32))
set_actor_root_state_tensor_indexed(sim, states, indices_gym, 3)
```

This doesn't seem like an issue, but `indices_torch.to(torch.int32)` will be garbage collected since there is no reference to it. This is because gymtorch.unwreap_tensor does not participate in the reference counting. 


Instead, we should do the following:

```python
indices_torch = torch.LongTensor([0, 17, 42])
indices_torch32 = indices_torch.to(torch.int32) # <--- this reference will keep the tensor alive
set_actor_root_state_tensor_indexed(sim, states, gymtorch.unwrap_tensor(indices_torch32), 3)
```

now `indices_torch.to(torch.int32)` will not be garbage collected since `indices_torch32` is referencing it.

### 4.6 Limitations

Only supported for PhysX backend.

A tensor "setter" function should only be called once per step such as `set_actor_root_state_tensor`. We can call other setter functions, but only once per function.

```python
gym.set_actor_root_state_tensor_indexed(sim, root_states, indices1, n1)
gym.set_actor_root_state_tensor_indexed(sim, root_states, indices2, n2)
```

this is no-no.


Also only call `refresh` functions once per step, and it should be called before any "setter" functions.

If we call refresh after, we will only get stale data in the next loop. Keep that in mind.

---
## 5. Force Sensors

### 5.1 Rigid Body Force Sensors
We can attach force sensors to rigid bodies to measure forces and torques. The readings are net forces and torques acting on parent body which include external forces, contact forces, and internal forces (joint drives). Body resting on ground plane will have a net force of zero.

Force sensors are created on assets. To create force sensor, you specify rigid body index where sensor is attached and relative poes of the sensor with respect to body origin. 

```python
body_idx = gym.find_actor_rigid_body_index(asset, "bodyName")

sensor_pose1 = gymapi.Transform(gymapi.Vec3(0.2, 0.0, 0.0))
sensor_pose2 = gymapi.Transform(gymapi.Vec3(-0.2, 0.0, 0.0))

sensor_idx1 = gym.create_asset_force_sensor(asset, body_idx, sensor_pose1)
sensor_idx2 = gym.create_asset_force_sensor(asset, body_idx, sensor_pose2)
```

You can also pass additional properties to sensor:

```python
sensor_props = gymapi.ForceSensorProperties()
sensor_props.enable_forward_dynamics_forces = True
sensor_props.enable_constraint_solver_forces = True 
sensor_props.use_world_frame = True

sensor_idx = gym.create_asset_force_sensor(asset, body_idx, sensor_pose, sensor_props)
```

After constructing these guys, you can access individual force sensors using index:

```python
actor_handle = gym.create_actor(env, asset, ...)
num_sensors  = gym.get_actor_force_sensor_count(env, actor_handle)
for i in range(num_sensors):
    sensor = gym.get_actor_force_sensor(env, actor_handle, i)
```

During simulation, you can query latest sensor readings:
```python
sensor_data = sensor.get_forces()
print(sensor_data.force) #Vec3
print(sensor_data.torque) #Vec3
```

Get total number of force sensors `gym.get_sim_force_sensor_count(sim)`.

### 5.1.1 Tensor API

We can get a Gym tensor descriptor for all force sensors in simulation using `gym.acquire_force_sensor_tensor(sim)`. 

```python
_fsdata = gym.acquire_force_sensor_tensor(sim)
fsdata = gymtorch.wrap_tensor(_fsdata)
```

The shape of this tensor is (num_force_sensors, 6) and data type is float32. The first 3 floats are force and last 3 are torque.

After each sim step, you can get latest sensor readings calling `gym.refresh_force_sensor_tensor(sim)`.


### 5.2 Joint Force Sensors

This is easier to do. Just call `enable_actor_dof_force_sensors` after creating the actor. 

```python
actor = gym.create_actor(env, ...)
gym.enable_actor_dof_force_sensors(env, actor)
```

During simulation, we retrieve forces using `get_actor_dof_forces`. 

```python
forces = gym.get_actor_dof_forces(env, actor_handle)
```

This returns numpy array of total forces acting on this actor's DOFF.

These forces will be zero if we do not enable joint force sensors.

### 5.2.1 Tensor API

We can also use the tensor API to get forces. 
```python
_forces = gym.acquire_dof_force_tensor(sim)
forces  = gymtorch.wrap_tensor(_forces)
```

This is a 1-dim tensor of float32 corresponding to each DOF in simulation. We can get latest sensor reading by calling `gym.refresh_dof_force_tensor(sim)`.

<b> DOF force sensors are only available for PhysX backend. </b>

---
## 6. Graphics and Camera Sensors


The point of this section is to show how to add camera sensors into our setup.

### 6.1 Camera Properties

All cameras can be created with parameters passed in GymCameraProperties.

```python
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

...

camera_props = gymapi.CameraProperties()
camera_props.horizontal_fov = 75.0 # degrees
camera_props.width = 1920
camera_props.height = 1080
viewer = gym.create_viewer(sim, camera_props)
```

### 6.2 Camera Sensors

Camera sensors are meant to simulate cameras as if they were an actual sensor. They are not meant to be used for rendering. 

```python
camera_props = gymapi.CameraProperties()
camera_props.width  = 128
camera_props.height = 128
camera_handle = gym.create_camera_sensor(env, camera_props)
```

We can set the camera location using `set_camera_location`:

```python
gym.set_camera_location(camera_handle, env, gymapi.Vec3(x,y,z), gymapi.Vec3(tx,ty,tz))
```

where `(x,y,z)` is the camera position and `(tx,ty,tz)` is position camera is looking at. All of this is in environment local coordinates. 

The method I like more is to specify `GymTransform`:

```python
transform = gymapi.Transform()
transform.p = (x,y,z)
transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
gym.set_camera_transform(camera_handle, env, body_handle, transform, gymapi.FOLLOW_TRANSFORM)
```

Last argument determines attachment behavior. 
- `gymapi.FOLLOW_POSITION` - camera maintains fixed offset from rigid body but will not rotate.
- `gymapi.FOLLOW_TRANSFORM` - camera maintains fixed transform relative to rigid body.

if we don't specify, the camera will be fixed in world space.


All cameras are rendered together in one API cal.

```python
gym.render_all_camera_sensors(sim)
```

We can access camera image using `get_camera_image`:

```python
color_image = gym.get_camera_image(sim, camera_handle, gymapi.IMAGE_COLOR)
depth_image = gym.get_camera_image(sim, camera_handle, gymapi.IMAGE_DEPTH)
seg_image = gym.get_camera_image(sim, camera_handle, gymapi.IMAGE_SEGMENTATION)
optical_flow_image = gym.get_camera_image(sim, camera_handle, gymapi.IMAGE_OPTICAL_FLOW)
```

One key thing is to keep the camera stuff in GPU. 

```python
camera_props = gymapi.CameraProperties()
camera_props.enable_tensors = True
cam_handle = gym.create_camera_sensor(env, camera_props)
```

```python
camera_tensor = gym.get_camera_image_gpu_tensor(sim, env, cam_handle, gymapi.IMAGE_COLOR)
torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
```

this allows us to get the camera image as a tensor and use it in PyTorch.


Here is a loop for cameras when using simulation in isaacgym:
```python
while True:

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    gym.start_access_image_tensors(sim)
    #
    # User code to digest tensors
    #
    gym.end_access_image_tensors(sim)
```
---
## Python API

- Checkout IsaacGym_Preview_TacSL_Package/isaacgym/docs/api/python/gym_py.html