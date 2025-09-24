---
layout: default
title: IsaacGymEnv Task Notes
parent: Isaac
nav_order: 9
---

# 7. Creating your own Task in IsaacGymEnv

All tasks in IsaacGymEnv inherit from an abstract base class `FactoryABCTask`. This class defines the structure and methods that any task must implement. Here's a breakdown of the key methods:

```python
class FactoryABCTask(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize instance variables. Initialize environment superclass."""
    @abstractmethod
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""
    @abstractmethod
    def _acquire_task_tensors(self):
        """Acquire tensors."""
    @abstractmethod
    def _refresh_task_tensors(self):
        """Refresh tensors."""
    @abstractmethod
    def pre_physics_step(self):
        """Reset environments. Apply actions from policy as controller targets. Simulation step called after this method."""
    @abstractmethod
    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""
    @abstractmethod
    def compute_observations(self):
        """Compute observations."""
    @abstractmethod
    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""
    @abstractmethod
    def _update_rew_buf(self):
        """Compute reward at current timestep."""
    @abstractmethod
    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
    @abstractmethod
    def reset_idx(self):
        """Reset specified environments."""
    @abstractmethod
    def _reset_franka(self):
        """Reset DOF states and DOF targets of Franka."""
    @abstractmethod
    def _reset_object(self):
        """Reset root state of object."""
    @abstractmethod
    def _reset_buffers(self):
        """Reset buffers."""
    @abstractmethod
    def _set_viewer_params(self):
        """Set viewer parameters."""
```

There is a lot more we need to provide to this task class compared to the environment class. All of the `_method` methods are all called within the task class.

The other methods are called by the script when it runs the task. For example, `post_physics_step` is called by the script extensively which computes observations and rewards every step. Most of the code past the task is abstracted which makes it difficult to understand how this codebase works.



