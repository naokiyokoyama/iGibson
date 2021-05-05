import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('path_to_weights')
parser.add_argument('desired_output_name')
args = parser.parse_args()

with open('Dockerfile') as f:
	data = f.read().splitlines()

print(f'Copying {args.path_to_weights} to docker_weights.pth...')
shutil.copyfile(args.path_to_weights, 'docker_weights.pth')
print('Done.')

data[4] = f"ADD docker_weights.pth /{args.desired_output_name}"
data[5] = f"ENV CHECKPOINT_PATH=/{args.desired_output_name}"


with open('Dockerfile', 'w') as f:
	f.write('\n'.join(data)+'\n')