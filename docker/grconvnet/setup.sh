#!/bin/bash

# create conda env
echo New Environment Name:
read envname

conda create -n $envname -y -q

eval "$(conda shell.bash hook)"
conda activate $envname

# Set Channel vars
conda config --add channels conda-forge
conda config --set channel_priority strict

# install mamba
conda install mamba -c conda-forge -y -q

mamba install python=3.9 -y -q


# install alr simulation framework, can be removed when we only need the service
mamba install -c conda-forge pybullet pyyaml scipy opencv pinocchio matplotlib gin-config gym -y -q
mamba install -c conda-forge scikit-learn addict pandas plyfile tqdm -y -q
mamba install -c open3d-admin open3d -y -q
mamba install -c conda-forge imageio -y -q
pip install mujoco
pip install git+https://github.com/ALRhub/SimulationFramework
# cd `dirname "$BASH_SOURCE"` && pip install -e .

# exit 0
