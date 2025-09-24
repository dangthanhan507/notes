---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
parent: Kuka FRI Notes
title: Code
math: katex
---

# Writing code

So there's multiple ways we can work with the current FRI connection to the Kuka Sunrise controller. At the end of the day, FRI setup between the computer and controller runs a UDP connection that sends LCM-wrapped packets. 

While lots of my own personal code uses Drake, you can bypass this completely by sending the LCM messages directly! This means that the only time you're using drake is to run the network driver (connection) which is completely separate from your own personal code.

# Running Network driver

Prerequisites:
- You have bazel!

```bash
$ bazel build
$ cd drake-iiwa-driver/bazel-bin/kuka-driver
$ ./kuka_driver --fri_port [insert port number here]
```

If anything weird is going on, just ctrl+c and re-run the driver :).

Make sure you are running the KukaFRIPositionController on the Kuka pendant!

## Using Drake (Python)

We need these python packages
- drake
- manipulation

Technically, we don't need `manipulation` package. However, it writes up a nice manipulation station code that allows us to work with the Kuka robot hardware. 

### YAML Configuration

We can write up the hardware configuration below. Here, we can write up the base transform of the Kuka robot, the default joint positions, and which FRI mode to use (position, impedance, torque).

Here is an example yaml file

```yaml
# iiwa_standard_setup.yaml
Demo:
  directives:
    # Add iiwa
        - add_model:
            name: iiwa
            file: package://drake/manipulation/models/iiwa_description/urdf/iiwa14_polytope_collision.urdf
            default_joint_positions:
                iiwa_joint_1: [0]
                iiwa_joint_2: [0]
                iiwa_joint_3: [0]
                iiwa_joint_4: [0]
                iiwa_joint_5: [0]
                iiwa_joint_6: [0]
                iiwa_joint_7: [0]

        - add_weld:
            parent: world
            child: iiwa::base
  model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
```

If we want to interface this in Drake, here's the code to do so:
```python
from manipulation.station import MakeHardwareStation, MakeHardwareStationInterface, load_scenario
from pydrake.all import (
    StartMeshcat,
)

meshcat = StartMeshcat()
scenario = load_scenario(filename="iiwa_standard_setup.yaml", scenario_name="Demo")
manipulation_station = MakeHardwareStationInterface(
            scenario,
            meshcat=meshcat, # can also be None
            package_xmls=[package_file]
)
# manipulation_station is now a System Block with inputs and outputs
# by default, its name is "iiwa", 
#  so accessing the ports is "iiwa.[insert name]"

'''
manipulation_station Inputs:
- iiwa.position: float32[7] (joint position targets)
- iiwa.feedforward_torque: float32[7] (joint feedforward torques)
manipulation_station Outputs:
- iiwa.position_measured: float32[7] (joint position measurements)
- iiwa.position_commanded: float32[7] (joint positions commanded)

For a more extensive list check out ManipulationStation in drake:
- https://drake.mit.edu/doxygen_cxx/classdrake_1_1examples_1_1manipulation__station_1_1_manipulation_station.html
'''
```

```python
# NOTE: re-using variables from above python snippet
# NOTE: this is code to make kuka go to a specific joint position
from pydrake.all import (
    DiagramBuilder,
    Simulator,
    ConstantVectorSource,
    RigidTransform
)
import numpy as np

DESIRED_JOINT = np.array([0.0, np.pi/6, 0.0, -80*np.pi/180, 0.0, np.pi/6, 0.0])
builder = DiagramBuilder()
manipulation_station_block = builder.AddSystem(manipulation_station)
desired_joint_block = builder.AddSystem(ConstantVectorSource(DESIRED_JOINT))
builder.Connect(
    desired_joint_block.get_output_port(),
    manipulation_station_block.get_input_port("iiwa.position")
)
diagram = builder.Build()

'''
ASCII Art of current Diagram we made:
        ------------------------
------> | manipulation_station | -------> FRI
|       ------------------------
|
| 
|
|               -----------------------
----------------| desired_joint_block |
DESIRED_JOINT   -----------------------
'''


simulator = Simulator(diagram)
simulator.AdvanceTo(30.0)
```

In order to do more complicated things in Drake, we need to use more of the drake library and have a deeper understanding of how their Diagrams/Systems work. That's a lot to ask for the reader right now, so this is the barebones example on how to use this for hardware.

## Using raw LCM

UNDER CONSTRUCTION