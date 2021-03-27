from gibson2.envs.igibson_env import iGibsonEnv
from habitat_cont.model import PointNavResNetAgent

import argparse
import time
import numpy as np

import cv2

WEIGHTS_PATH = '/coc/pskynet3/nyokoyama3/aug26/hablabhotswap/d4_noslide_92.json'
# WEIGHTS_PATH = '/coc/pskynet3/nyokoyama3/aug26/hablabhotswap/gaussian_noslide_30deg_63_skyfail.json'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]'
    )
    parser.add_argument(
        '--save_fov',
        '-s',
        action='store_true',
        help='whether to store the video or not.'
    )
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

        if args.save_fov:
            four_cc = cv2.VideoWriter_fourcc(*'MP4V')    
            out_vid = cv2.VideoWriter(
                f'agent_fov_ep_{episode}.mp4',
                four_cc,
                10.0,
                (env.config['image_width']*2, env.config['image_height'])
            )

        print('Episode: {}'.format(episode))
        start = time.time()
        state = env.reset()
        for steppy in range(5000):  # 10 seconds
            if steppy % 10 == 0:
                depth, rgb = state['depth'], state['rgb']
                dist, heading = state['task_obs'][:2]
                pointgoal_with_gps_compass = np.array([dist, heading])
                lv, av = agent.act(
                    depth                      = depth,
                    pointgoal_with_gps_compass = pointgoal_with_gps_compass
                )
                action = np.array([lv, av])
                print(f'dist: {dist:0.3f}\tang: {np.rad2deg(heading):0.3f}\tlv: {lv*0.25:0.3f}\tav: {av*30:0.3f}')
                if args.save_fov:
                    depth_mono_256 = np.array(depth*255, dtype=np.uint8)
                    depth_256 = cv2.cvtColor(depth_mono_256, cv2.COLOR_GRAY2BGR)
                    rgb_256 = np.array(rgb*255, dtype=np.uint8)
                    bgr_256 = cv2.cvtColor(rgb_256, cv2.COLOR_RGB2BGR)
                    frame = np.hstack([bgr_256, depth_256])
                    out_vid.write(frame)

            state, reward, done, _ = env.step(action)
            if done:
                break
        agent.reset()
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
        if args.save_fov:
            out_vid.release()
    env.close()
