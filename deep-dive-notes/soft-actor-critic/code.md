---
layout: default
title: SAC Code
parent: Soft Actor Critic
mathjax: true
tags: 
  - latex
  - math
has_children: true
---
# Soft Actor Critic

We will use rl_games to understand Soft Actor Critic (SAC) implementation.

The SAC algorithm is a `BaseAlgorithm` or `BasePlayer` in rl_games. 

At a high level, SAC is deployed similarly to other algorithms in rl_games like `a2c_continuous` or `a2c_discrete`. The main implementation should all be inside `sac_agent.py`.

`BaseAlgorithm` has a few key functions we should be interested in investigating... `train()` and `train_epoch()`.

```python
def train(self):
    self.init_tensors()
    self.algo_observer.after_init(self)
    total_time = 0
    self.obs = self.env_reset()

    while True:
        self.epoch_num += 1
        step_time, play_time, update_time, epoch_total_time, actor_losses, entropies, alphas, alpha_losses, critic1_losses, critic2_losses = self.train_epoch()

        total_time += epoch_total_time

        curr_frames = self.num_frames_per_epoch
        self.frame += curr_frames

        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / play_time
        fps_total = curr_frames / epoch_total_time

        if self.game_rewards.current_size > 0:
            mean_rewards = self.game_rewards.get_mean()
            mean_lengths = self.game_lengths.get_mean()

            checkpoint_name = self.config['name'] + '_ep_' + str(self.epoch_num) + '_rew_' + str(mean_rewards)

            should_exit = False

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

            if mean_rewards > self.last_mean_rewards and self.epoch_num >= self.save_best_after:
                print('saving next best rewards: ', mean_rewards)
                self.last_mean_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, self.config['name']))
                if self.last_mean_rewards > self.config.get('score_to_win', float('inf')):
                    print('Maximum reward achieved. Network won!')
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    should_exit = True

            if self.epoch_num >= self.max_epochs and self.max_epochs != -1:
                if self.game_rewards.current_size == 0:
                    print('WARNING: Max epochs reached before any env terminated at least once')
                    mean_rewards = -np.inf

                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(self.epoch_num) \
                    + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                print('MAX EPOCHS NUM!')
                should_exit = True

            if self.frame >= self.max_frames and self.max_frames != -1:
                if self.game_rewards.current_size == 0:
                    print('WARNING: Max frames reached before any env terminated at least once')
                    mean_rewards = -np.inf

                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                    + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                print('MAX FRAMES NUM!')
                should_exit = True

            update_time = 0

            if should_exit:
                return self.last_mean_rewards, self.epoch_num
```