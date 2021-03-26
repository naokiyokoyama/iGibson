from gibson2.envs.igibson_env import iGibsonEnv
from habitat_cont.model import PointNavResNetAgent

import argparse
import time
import numpy as np

# WEIGHTS_PATH = '/coc/pskynet3/nyokoyama3/aug26/hablabhotswap/d4_noslide_92.json'
WEIGHTS_PATH = '/coc/pskynet3/nyokoyama3/aug26/hablabhotswap/gaussian_noslide_30deg_63_skyfail.json'

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

    agent = PointNavResNetAgent(WEIGHTS_PATH)

    env = iGibsonEnv(config_file=args.config,
                     mode=args.mode,
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 240.0)

    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        state = env.reset()
        for steppy in range(5000):  # 10 seconds
            # action = env.action_space.sample()
            # print('action', action)
            if steppy % 10 == 0:
                depth = state['depth']
                # depth = np.expand_dims(state["depth"][0] / 255, axis=2)
                dist, heading = state['task_obs'][:2]
                adj_heading = heading#+np.pi/2+np.pi
                if adj_heading > np.pi:
                    adj_heading -= 2*np.pi
                elif adj_heading < -np.pi:
                    adj_heading += 2*np.pi
                pointgoal_with_gps_compass = np.array([dist, adj_heading])
                lv, av = agent.act(
                    depth                      = depth,
                    pointgoal_with_gps_compass = pointgoal_with_gps_compass
                )
                action = np.array([lv, av])
                print(f'dist: {dist:0.3f}\tang: {np.rad2deg(adj_heading):0.3f}\tlv: {lv*0.25:0.3f}\tav: {av*30:0.3f}')

            state, reward, done, _ = env.step(action)
            # print('action', action)
            # print('state', state['task_obs'])
            if done:
                break
        agent.reset()
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
