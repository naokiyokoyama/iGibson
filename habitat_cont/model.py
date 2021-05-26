from habitat_baselines.rl.ddppo.policy.resnet_policy import (
    PointNavResNetPolicy,
)
from habitat.core.spaces import ActionSpace, EmptySpace

import json
import numpy as np
import torch
import time
import os

from gym import spaces
from gym.spaces import Dict as SpaceDict
from collections import OrderedDict, defaultdict

DEVICE = torch.device("cuda")

def to_tensor(v): # DON'T CHANGE
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

class PointNavResNetAgent:
    def __init__(self, weights_path, gaussian=True):
        self.model = self.load_model(weights_path)
        self.reset()

    def load_model(self, weights_path):

        checkpoint = torch.load(
            weights_path, map_location="cpu"
        )

        config = checkpoint['config']

        depth_256_space = SpaceDict({
            'depth': spaces.Box(low=0., high=1., shape=(180,320,1)),
            'pointgoal_with_gps_compass': spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        })

        if config.RL.POLICY.action_distribution_type == "gaussian":
            action_space = ActionSpace(
                {
                    "linear_velocity": EmptySpace(),
                    "angular_velocity": EmptySpace(),
                }
            )
            self.action_distribution = 'gaussian'
        else:
            self.num_actions = len(
                config.TASK_CONFIG.TASK.ACTIONS.VELOCITY_CONTROL.get(
                    'DISCRETE_ACTIONS',
                    list(range(4))
                )
            )
            action_space = spaces.Discrete(self.num_actions)
            self.action_distribution = 'categorical'

            if self.num_actions == 9:
                self.discrete_actions = (
                    config.TASK_CONFIG.TASK.ACTIONS.VELOCITY_CONTROL.DISCRETE_ACTIONS
                )

        model = PointNavResNetPolicy(
            observation_space=depth_256_space,
            action_space=action_space,
            hidden_size=512,
            rnn_type=config.RL.DDPPO.rnn_type,  
            num_recurrent_layers=2,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs=False,
            force_blind_policy=False,
            action_distribution_type=self.action_distribution,
        )
        model.to(DEVICE)

        # Load weights
        data_dict = checkpoint['state_dict']
        model.load_state_dict(
            {
                k[len("actor_critic.") :]: torch.tensor(v)
                for k, v in data_dict.items()
                if k.startswith("actor_critic.")
            }
        )

        return model

    def reset(self):

        self.test_recurrent_hidden_states = torch.zeros(
            1, # self.config.NUM_ENVIRONMENTS,
            self.model.net.num_recurrent_layers,
            512, # ppo_cfg.hidden_size,
            device=DEVICE,
        )
        self.not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=DEVICE)

        if self.action_distribution == 'gaussian':
            self.prev_actions = torch.zeros(1, 2, device=DEVICE)
        else:
            self.prev_actions = torch.zeros(
                1, 1, dtype=torch.long, device=DEVICE
            )

    def act(self, depth, pointgoal_with_gps_compass):

        observations = {
            'depth': depth,
            'pointgoal_with_gps_compass': pointgoal_with_gps_compass
        }

        batch = defaultdict(list)

        for sensor in observations:
            batch[sensor].append(to_tensor(observations[sensor]))

        for sensor in batch:
            batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                device=DEVICE, dtype=torch.float
            )

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.model.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )
        self.prev_actions.copy_(actions)
        self.not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=DEVICE)

        actions = actions.squeeze()

        max_linear_speed = 1.0
        max_angular_speed = 1.0
        if self.action_distribution == 'gaussian':
            move_amount = torch.clip(actions[0], min=-1, max=1).item()
            turn_amount = torch.clip(actions[1], min=-1, max=1).item()
            move_amount = (move_amount+1.)/2.
        elif self.num_actions == 4:
            action_index = actions.item()
            discrete_actions = [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ]
            move_amount, turn_amount = discrete_actions[action_index]
        elif self.num_actions == 9:
            action_index = actions.item()
            move_amount, turn_amount = self.discrete_actions[action_index]
        return move_amount, turn_amount
