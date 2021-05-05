# Parent Docker image
FROM gibsonchallenge/gibson_challenge_2021:latest

# Copy weights for the agent. These MUST be lines 5 and 6 of this file.
ADD docker_weights.pth /sn_2_49.pth
ENV CHECKPOINT_PATH=/sn_2_49.pth

# Add gibson conda env to the PATH
ENV PATH /miniconda/envs/gibson/bin:$PATH

# cd into /opt
WORKDIR /opt

# Clone Habitat
RUN git clone --branch igibson_challenge https://github.com/naokiyokoyama/habitat-lab /opt/habitat-lab

# Install additional dependencies
RUN pip install -r /opt/habitat-lab/requirements.txt
WORKDIR /opt/habitat-lab
RUN python setup.py develop --all

# Download agent.py from repo
RUN apt install wget
RUN wget https://raw.githubusercontent.com/naokiyokoyama/iGibson/team_cvmlp/agent.py -O /agent.py

# Create '/submission.sh' script
RUN echo 'python /agent.py' > /submission.sh

WORKDIR /
