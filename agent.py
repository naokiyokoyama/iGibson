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

from gibson2.challenge.challenge import Challenge

DEVICE = torch.device("cuda")
CHECKPOINT_PATH = os.environ['CHECKPOINT_PATH']

def to_tensor(v): # DON'T CHANGE
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

class PointNavResNetAgent:
    def __init__(self, weights_path):

        self.GAUSSIAN, self.DISCRETE_4, self.DISCRETE_6 = False, False, False
        if 'd4' in weights_path:
            self.DISCRETE_4 = True
        elif 'd6' in weights_path:
            self.DISCRETE_6 = True
        elif 'gaussian' in weights_path or 'sn_' in weights_path:
            self.GAUSSIAN = True

        self.model = self.load_model(weights_path)
        self.reset()

    def load_model(self, weights_path):

        depth_256_space = SpaceDict({
            # 'depth': spaces.Box(low=0., high=1., shape=(256,256,1)),
            'depth': spaces.Box(low=0., high=1., shape=(180,320,1)),
            'pointgoal_with_gps_compass': spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        })

        if self.GAUSSIAN:
            action_space = ActionSpace(
                {
                    "linear_velocity": EmptySpace(),
                    "angular_velocity": EmptySpace(),
                }
            )
            action_distribution = 'gaussian'
            dim_actions = 2
        elif self.DISCRETE_4:
            action_space = spaces.Discrete(4)
            action_distribution = 'categorical'
            dim_actions = 4
        elif self.DISCRETE_6:
            action_space = spaces.Discrete(6)
            action_distribution = 'categorical'
            dim_actions = 6

        model = PointNavResNetPolicy(
            observation_space=depth_256_space,
            action_space=action_space,
            hidden_size=512,
            rnn_type='LSTM',
            num_recurrent_layers=2,
            backbone='resnet50',
            normalize_visual_inputs=False,
            force_blind_policy=False,
            action_distribution_type=action_distribution,
        )
        model.to(DEVICE)

        data_dict = torch.load(
            weights_path, map_location="cpu"
        )['state_dict']

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

        if self.GAUSSIAN:
            self.prev_actions = torch.zeros(1, 2, device=DEVICE)
        else:
            self.prev_actions = torch.zeros(1, 1, device=DEVICE)

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

        if self.GAUSSIAN:
            move_amount = torch.clip(actions[0], min=-1, max=1).item()
            turn_amount = torch.clip(actions[1], min=-1, max=1).item()
            move_amount = (move_amount+1.)/2.

        elif self.DISCRETE_4:
            action_index = actions.item()
            move_amount, turn_amount = 0,0
            if action_index == 0: # STOP
                print('[STOP HAS BEEN CALLED]')
            elif action_index == 1: # Move FWD
                move_amount = max_linear_speed
            elif action_index == 2: # LEFT
                turn_amount = max_angular_speed
            else: # RIGHT
                turn_amount = -max_angular_speed

        elif self.DISCRETE_6:
            action_index = actions.item()
            move_amount, turn_amount = 0,0
            if action_index == 0:
                move_amount = max_linear_speed
                turn_amount = -max_angular_speed
            elif action_index == 1:
                turn_amount = -max_angular_speed
            elif action_index == 2:
                move_amount = max_linear_speed
            elif action_index == 3:
                print('[STOP HAS BEEN CALLED]')
            elif action_index == 4:
                move_amount = max_linear_speed
                turn_amount = max_angular_speed
            elif action_index == 5: # 
                turn_amount = max_angular_speed

        return move_amount, turn_amount

class PointNavResNetAgentV2(PointNavResNetAgent):
    def act(self, observations):
        depth, rgb = observations['depth'], observations['rgb']
        dist, heading = observations['task_obs'][:2]
        pointgoal_with_gps_compass = np.array([dist, heading])

        lv, av = super().act(depth, pointgoal_with_gps_compass)

        action = np.array([lv, av])

        return action


def main():
    agent = PointNavResNetAgentV2(CHECKPOINT_PATH)
    challenge = Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()