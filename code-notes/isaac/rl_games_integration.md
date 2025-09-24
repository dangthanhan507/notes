---
layout: default
title: RL Games Coupling
parent: Isaac
nav_order: 10
---

# RL Games Coupling

In IsaacGymEnvs `train.py`, we use `Runner`, `model_builder`, `env_configurations`, and `vecenv`. 

Right now it seems `vecenv`, `model_builder`, and `env_configurations` are used for flags/configurations, while `Runner has most of the logic for training and deployment.


# RL_GAMES: Runner

Runner is made in `rl_games.torch_runner.py`.

First observation we make is in the `__init__` method,

```python
class Runner:
    def __init__(self, algo_observer=None):

        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs : a2c_discrete.DiscreteA2CAgent(**kwargs)) 
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs : players.PpoPlayerContinuous(**kwargs))
        self.player_factory.register_builder('a2c_discrete', lambda **kwargs : players.PpoPlayerDiscrete(**kwargs))
        self.player_factory.register_builder('sac', lambda **kwargs : players.SACPlayer(**kwargs))

        self.algo_observer = algo_observer if algo_observer else DefaultAlgoObserver()
        torch.backends.cudnn.benchmark = True
```

this gives us a clue that there is a separation between "algo" and "player". Additionally, there were comments in this class that suggests `Runner` also logs metrics for training and evaluation.

Take note that `algo_observer` is the observer for logging purposes.

Next thing `Runner` does in training script is to run `runner.load(rlg_config_dict)`. 

```python
def load(self, yaml_config):
    config = deepcopy(yaml_config)
    self.default_config = deepcopy(config['params'])
    self.load_config(params=self.default_config)

def load_config(self, params):
    # NOTE: params is from a config file (.yaml) 

    self.seed = params.get('seed', None)
    if self.seed is None:
        self.seed = int(time.time())

    ... # there was code here for multi_gpu

    print(f"self.seed = {self.seed}")

    self.algo_params = params['algo']
    self.algo_name = self.algo_params['name']
    self.exp_config = None

    ... # there was code here for seed handling

    config = params['config']
    config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**config['reward_shaper'])
    if 'features' not in config:
        config['features'] = {}
    config['features']['observer'] = self.algo_observer
    self.params = params
```

Finally `Runner` runs `runner.play()` which is the main loop for training and evaluation.

```python
def run(self, args):
    if args['train']:
        self.run_train(args)
    elif args['play']:
        self.run_play(args)
    else:
        self.run_train(args)
# Lets look at run_play
def run_play(self, args):
    player: players.PpoPlayerContinuous = self.create_player()
    _restore(player, args) # load
    _override_sigma(player, args) # override parameters in network labelled sigma
    player.run()
def create_player(self):
    # remember self.player_factory from __init__!.
    return self.player_factory.create(self.algo_name, params=self.params)
class ObjectFactory:
    def __init__(self):
        self._builders = {}
    def register_builder(self, name, builder):
        self._builders[name] = builder
    ...
    def create(self, name, **kwargs):
        # NOTE: name we use is a2c_continuous
        builder = self._builders.get(name) 
        if not builder:
            raise ValueError(name)
        return builder(**kwargs) # this is players.PpoPlayerContinuous
```

Since we are initializing the `players.PpoPlayerContinuous` class, we can look at its `__init__` method (which is called in `create_player`).

```python
class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
```

For training, we get our configs from a yaml file like `TacSLTaskInsertionPPO_LSTM_dict_AAC.yaml` which has all of the stuff rl_games cares about.

Based on this `__init__` there is no reference of any of the `IsaacGymEnvs` configs. It's just from the train yaml file like above. In order to load up the rl_games network and use it, we only need this yaml file.

As we have discussed before, `TacSLTaskInsertionPPO_LSTM_dict_AAC.yaml` is a yaml file we would use to load and build the network in rl_games. Where is this code referenced in `IsaacGymEnvs`? It is in `train.py`. It is actually the `rlg_config_dict` file that we pass.

**TLDR**: `Runner` is the main class that we want to work with. We should get all the parameters we need for our network from a yaml file like `TacSLTaskInsertionPPO_LSTM_dict_AAC.yaml`. This yaml file is passed to `Runner` as `rlg_config_dict` in `train.py`.

# RL_GAMES: environment

How does `rl_games` actually use the environment from IsaacGymEnvs?

In `train.py` for IsaacGymEnvs, we have the following code:

```python
from rl_games.common import env_configurations, vecenv
def create_isaacgym_env(**kwargs):
    envs = isaacgymenvs.make(
        cfg.seed, 
        cfg.task_name, 
        cfg.task.env.numEnvs, 
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
        **kwargs,
    )
    ... # capture_video code here
    return envs

env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
})
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
```

This `env_configurations` works really like how `algo_factory` works where it's a dictionary of environment configurations.

This is the same for `vecenv`. You can consider `vecenv` and `env_configurations` as globals that `rl_games` will use down the line. 

We actually can see this by looking at the parent class of `PpoPlayerContinuous`, which is `BasePlayer`:

```python
class BasePlayer(object):
    def __init__(self, params):
        # params is from that .yaml file we mentioned
        self.config = config = params['config']
        self.load_networks(params)
        self.env_name = self.config['env_name']
        self.player_config = self.config.get('player', {})
        self.env_config = self.config.get('env_config', {})
        self.env_config = self.player_config.get('env_config', self.env_config)
        self.env_info = self.config.get('env_info')
        ... # extra configs
        
        if self.env_info is None:
            use_vecenv = self.player_config.get('use_vecenv', False)
            if use_vecenv:
                print('[BasePlayer] Creating vecenv: ', self.env_name)
                self.env = vecenv.create_vec_env(
                    self.env_name, self.config['num_actors'], **self.env_config)
                self.env_info = self.env.get_env_info()
            else:
                print('[BasePlayer] Creating regular env: ', self.env_name)
                self.env = self.create_env()
                self.env_info = env_configurations.get_env_info(self.env)
        else:
            self.env = config.get('vec_env')
        # NOTE: vecenv and env_configurations are globals updated
        # we know how to look up the configs because of the params
```

We can see `self.env` is created and we use it during `runner.run`. 

**TLDR**: So if we want to create a new environment, we just have to register it in `env_configurations` and `vecenv` like we did above.

# IsaacGymEnvs: Things needed in environment

For IsaacGymEnvs, we create an environment by calling the following

```python
isaacgym_task_map = {
    "AllegroHand": AllegroHand,
    "AllegroKuka": resolve_allegro_kuka,
    "AllegroKukaTwoArms": resolve_allegro_kuka_two_arms,
    "AllegroHandManualDR": AllegroHandDextremeManualDR,
    "AllegroHandADR": AllegroHandDextremeADR,
    "Ant": Ant,
    "Anymal": Anymal,
    "AnymalTerrain": AnymalTerrain,
    "BallBalance": BallBalance,
    "Cartpole": Cartpole,
    "FactoryTaskGears": FactoryTaskGears,
    "FactoryTaskInsertion": FactoryTaskInsertion,
    "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
    "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
    "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
    "IndustRealTaskPegsInsert": IndustRealTaskPegsInsert,
    "IndustRealTaskGearsInsert": IndustRealTaskGearsInsert,
    "FrankaCabinet": FrankaCabinet,
    "FrankaCubeStack": FrankaCubeStack,
    "Humanoid": Humanoid,
    "HumanoidAMP": HumanoidAMP,
    "Ingenuity": Ingenuity,
    "Quadcopter": Quadcopter,
    "ShadowHand": ShadowHand,
    "TacSLTaskInsertion": TacSLTaskInsertion,
    "AmazonTaskInsertion": AmazonTaskInsertion,
    "Trifinger": Trifinger,
}
...
env = isaacgym_task_map[task_name](
    cfg=task_config,
    rl_device=_rl_device,
    sim_device=_sim_device,
    graphics_device_id=graphics_device_id,
    headless=headless,
    virtual_screen_capture=virtual_screen_capture,
    force_render=force_render,
)
```

Out of this `env` needs the following properties:
```python
env.observation_space
env.action_space 
env.agents # 1
env.value_size # 1
env.get_number_of_agents # optional

obs = env.reset()
obs, rewards, dones, infos = env.step(actions)
```