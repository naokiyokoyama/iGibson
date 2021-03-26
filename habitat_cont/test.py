from gibson2.envs.igibson_env import iGibsonEnv

import argparse
import time
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config,
                     mode=args.mode,
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            depth = state['depth']
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
