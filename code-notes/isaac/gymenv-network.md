---
layout: default
title: IsaacGymEnv Network Notes
parent: Isaac
nav_order: 5
---

# 3. IsaacGymEnv Network Notes

This is a tougher set of notes because the codebase seemingly disconnects the tacsl code from the network building code. Additionally, rl_games abstracts away a lot of details, but since we want to add our own networks, we will have to remove that abstraction.

**GOAL:** Understand enough to add PointNet to the network code.

## 3.1 How does input work?

### 3.1.1 Input Preprocessing Specification

We can work from the training script commandline arguments to see how the network is built. Here's an example for tactile training. 

```python
task.env.obsDims={
    ee_pos:[3],
    ee_quat:[4],
    socket_pos:[3],
    socket_quat:[4],
    dof_pos:[9]
}
task.env.obsDims={
    left_tactile_camera_taxim:[80,60,3],
    right_tactile_camera_taxim:[80,60,3],
    object_pcd:[500,3]
}
task.env.stateDims={
    ee_pos:[3],
    ee_quat:[4],
    plug_pos:[3],
    plug_quat:[4],
    socket_pos_gt:[3],
    socket_quat:[4],
    dof_pos:[9],
    ee_lin_vel:[3],
    ee_ang_vel:[3],
    plug_socket_force:[3],
    plug_left_elastomer_force:[3],
    plug_right_elastomer_force:[3]
}
train.params.network.input_preprocessors={
    ee_pos:{},
    ee_quat:{},
    socket_pos:{},
    socket_quat:{},
    dof_pos:{}
}
train.params.config.central_value_config.network.input_preprocessors={
    ee_pos:{},
    ee_quat:{},
    plug_pos:{},
    plug_quat:{},
    socket_pos_gt:{},
    socket_quat:{},
    dof_pos:{},
    ee_lin_vel:{},
    ee_ang_vel:{},
    plug_socket_force:{},
    plug_left_elastomer_force:{},
    plug_right_elastomer_force:{}
}
train.params.network.input_preprocessors={
    left_tactile_camera_taxim:{
        cnn:{
            type: conv2d_spatial_softargmax,
            activation: relu,
            initializer: {name: default},
            regularizer: {name: 'None'},
            convs: [{filters:32, kernel_size:8, strides:2, padding:0},
                    {filters:64, kernel_size:4, strides:1, padding:0},
                    {filters:64, kernel_size:3, strides:1, padding:0}]
        }
    }
    right_tactile_camera_taxim:{
        cnn:{
            type: conv2d_spatial_softargmax,
            activation: relu,
            initializer: {name: default},
            regularizer: {name: 'None'},
            convs: [{filters:32, kernel_size:8, strides:2, padding:0},
                    {filters:64, kernel_size:4, strides:1, padding:0},
                    {filters:64, kernel_size:3, strides:1, padding:0}]
        }
    }
}
```

We can make multiple observations here:
1. We split between observation dimensions and state dimensions.
2. State dimensions will be used for the central value network.
3. Observation dimensions will be used for the policy network (keys in `train.params.network.input_preprocessors`).
4. All of the architectural details for observations are in the `input_preprocessors` dictionary.
5. Specifying nothing `{}` means that input is passed through without modification.
6. The `cnn` key specifies a convolutional neural network with the given parameters.
7. Based on `isaacgymenvs/cfg/train/AllegroHandPPO.yaml`, we can also use `mlp` as a preprocessor too.

<!-- <b> NOTE: </b> after trying to dig into how input_preprocessing works, it seems that there is a lot of abstraction going on. In order to understand where these commands are parsed and how we know what observation head to use, we have to understand how `NetworkBuilder` from `rl_games` works and is being utilized.  -->

### 3.1.2 Input Preprocessing Code

IsaacGymEnvs uses `A2CBuilder` which it created which inherits from `NetworkBuilder` from `rl_games`. Inside of `A2CBuilder`, it also creates `Network` which inherits from `NetworkBuilder.BaseNetwork` from `rl_games` also.

It seems that `Network` is what we want to modify:

```python
class Network(NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):

        # Network structure:
        # obs is a dictionary, each gets processed according to the network in input_params.
        # Then features from each are flattened, concatenated, and go through an MLP.
        # for example
        #    camera -> CNN ->
        #                    |-> MLP -> action
        # joint_pos -> MLP ->

        ...
        self.load(params)
        ...

        for input_name, input_config in self.input_params.items():
            has_cnn = 'cnn' in input_config
            has_mlp = 'mlp' in input_config

            member_input_shape = torch_ext.shape_whc_to_cwh(input_shape[input_name])
            networks_actor = []
            networks_critic = []

            # add CNN head if it is part of the config
            if has_cnn:
                cnn_config = input_config['cnn']
                cnn_args = {
                    'ctype': cnn_config['type'],
                    'input_shape': member_input_shape,
                    'convs': cnn_config['convs'],
                    'activation': cnn_config['activation'],
                    'norm_func_name': self.normalization,
                }
                networks_actor.append(self._build_conv(**cnn_args))
                cnn_init = self.init_factory.create(**input_config['cnn']['initializer'])  # to be use later

                if self.separate:
                    networks_critic.append(self._build_conv(**cnn_args))
            # output shape from the CNN (left tactile + right tactile) probs 128 each
            next_input_shape = self._calc_input_size(
                member_input_shape, networks_actor[-1] if has_cnn else None
            )

            if has_mlp:
                mlp_config = input_config['mlp']
                ... # mlp_args stuff
                networks_actor.append(self._build_mlp(**mlp_args))
                if self.separate:
                    networks_critic.append(self._build_mlp(**mlp_args))
                next_input_shape = mlp_config['units'][-1] # add the output size of the mlp but states are low dim and no linear layers so just 256 + 23

            # mlp?

            self.input_networks_actor[input_name] = nn.Sequential(*networks_actor)
            self.input_networks_critic[input_name] = nn.Sequential(*networks_critic)
            input_out_size += next_input_shape
        in_mlp_shape = input_out_size
    def load(self, params):
        ...
        self.input_params = params['input_preprocessors'] # AN NOTE: where our command is processed
        ...
     
```

As we can see, it takes each observation taken by input_preprocessing and throws it into a CNN or MLP or None. You can track this by the `next_input_shape` variable which increases depending on whether it is using a CNN or MLP or None.

To better understand how to add our on observation head, we look at `_build_conv` which is inside rl_games:

```python
def _build_conv(self, ctype, **kwargs):
    print('conv_name:', ctype)

    if ctype == 'conv2d':
        return self._build_cnn2d(**kwargs)
    ... # other conv types

def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None,
                    add_spatial_softmax=False, add_flatten=False):
    in_channels = input_shape[0]
    layers = []
    for conv in convs:
        layers.append(conv_func(in_channels=in_channels, 
        out_channels=conv['filters'], 
        kernel_size=conv['kernel_size'], 
        stride=conv['strides'], padding=conv['padding']))
        conv_func=torch.nn.Conv2d
        act = self.activations_factory.create(activation)
        layers.append(act)
        in_channels = conv['filters']
        if norm_func_name == 'layer_norm':
            layers.append(torch_ext.LayerNorm2d(in_channels))
        elif norm_func_name == 'batch_norm':
            layers.append(torch.nn.BatchNorm2d(in_channels))
    if add_spatial_softmax:
        layers.append(SpatialSoftArgmax(normalize=True))
    if add_flatten:
        layers.append(torch.nn.Flatten())
    return nn.Sequential(*layers)
```

We use the arguments from the command as `__init__` arguments to the builder and then construct our observation head in a straightforward manner. We just have to make sure we return one `nn.Module` instead of many.

### 3.1.3 Adding your own Input Preprocessor

First you have to add `_build_pointnet` in the `rl_games` implementation along with `_build_conv` and `_build_mlp`. It's pretty straightforward.

Next, you add this to the `Network.__init__`:

```python
    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):

            # Network structure:
            # obs is a dictionary, each gets processed according to the network in input_params.
            # Then features from each are flattened, concatenated, and go through an MLP.
            # for example
            #    camera -> CNN ->
            #                    |-> MLP -> action
            # joint_pos -> MLP ->

            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            # self.actor_cnn = nn.Sequential()
            # self.critic_cnn = nn.Sequential()

            self.input_networks_actor = nn.ModuleDict()
            self.input_networks_critic = nn.ModuleDict()

            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            # size of the preprocessing of inputs before the main MLP
            input_out_size = 0

            for input_name, input_config in self.input_params.items():
                has_cnn = 'cnn' in input_config
                has_mlp = 'mlp' in input_config
                has_pnet = 'pnet' in input_config

                member_input_shape = torch_ext.shape_whc_to_cwh(input_shape[input_name])
                networks_actor = []
                networks_critic = []

                if has_pnet:
                    pnet_config = input_config['pnet']
                    pnet_args = {
                        'in_channel': pnet_config['in_channel'],
                        'out_channel': pnet_config['out_channel'],
                        'use_layernorm': False,
                        'final_norm': 'none',
                        'use_projection': True
                    }
                    networks_actor.append(self._build_pointnet(**pnet_args))
                    # TODO: make init
                    if self.separate:
                        networks_critic.append(self._build_pointnet(**pnet_args))
                    next_input_shape = pnet_config['out_channel']  # output shape from the pointnet
                ... # the other types
```

Next, you add this command to  your training script:

```python
task.env.obsDims={
    object_pcd:[500,3]
}
train.params.network.input_preprocessors={
    object_pcd:{
        pnet:{
            in_channel:3,
            out_channel:128
        }
    }
}
```

I've already included `object_pcd` in my previous observation notes.