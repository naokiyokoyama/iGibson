import os 
import numpy as np
import argparse

os.environ['CONFIG_FILE'] = '/coc/testnvme/nyokoyama3/gibson_challenge/iGibson/gibson2/examples/configs/locobot_interactive_nav.yaml'
os.environ['TASK'] = 'interactive'
os.environ['SPLIT'] = 'minival'
os.environ['EPISODE_DIR'] = '/coc/testnvme/nyokoyama3/gibson_challenge/iGibson/gibson2/data/episodes_data/interactive_nav'

from gibson2.challenge.challenge import Challenge

from habitat_cont.model import PointNavResNetAgent as PointNavResNetAgentOrig
WEIGHTS_PATH = 'weights/sn_1_49.pth'


class PointNavResNetAgent(PointNavResNetAgentOrig):
    def act(self, observations):
        depth, rgb = observations['depth'], observations['rgb']
        dist, heading = observations['task_obs'][:2]
        pointgoal_with_gps_compass = np.array([dist, heading])

        lv, av = super().act(depth, pointgoal_with_gps_compass)

        action = np.array([lv, av])

        return action

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--weights_path', default=WEIGHTS_PATH)
    args = parser.parse_args()

    agent = PointNavResNetAgent(args.weights_path)
    challenge = Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()
