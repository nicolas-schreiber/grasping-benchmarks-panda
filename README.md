# grasping-benchmarks-panda

## Essentials from the original README
Here the shortest version of instructions on how to use this modified version of the repo:

```bash
mkdir -p ~/catkin_ws/src

# Clone this Repo into the src folder
cd ~/catkin_ws/src
git clone git@github.com:nicolas-schreiber/grasping-benchmarks-panda.git

# Build the Repo (just creates the representations of the GraspRequest ROS Message and the GraspService)
# We prefer catkin build, but catkin_make should also work
catkin build
# or
# cd ~/catkin_ws
# catkin_make

# Build the docker containers
cd ~/catkin_ws/src/grasping-benchmarks-panda/docker
make USER_NAME=<your username here> dexnet

# Run the docker container
cd ..
bash run.sh nicolas_schreiber dexnet_container nicolas_schreiber/benchmark_dexnet

# ========= Further useful commands =============== 
# When running the docker container again you can simply call, this reuses the last docker container
bash run.sh nicolas_schreiber dexnet_container

# For any C++ or Python scripts to have access to the `BenchmarkGrasp.msg` the `GraspPlanner.srv` run following command:
source ~/catkin_ws/devel/setup.bash

# By calling this in a different terminal you can check what services are available:
rosservice list
```

So far, this repo includes support for:

| Algorithm | Documentation | Paper |
| --- | --- | --- |
**Dexnet** | [docs](https://berkeleyautomation.github.io/dex-net/)  | [paper](https://arxiv.org/pdf/1703.09312.pdf) |
**GPD** | [docs](https://github.com/atenpas/gpd) | [paper](https://arxiv.org/pdf/1706.09911.pdf) |
**Superquadrics-based grasp planner**  | [docs](https://github.com/robotology/superquadric-lib) | [paper](http://lornat75.github.io/papers/2017/vezzani-icra.pdf) |
**6DoF-GraspNet** | [docs](https://github.com/jsll/pytorch_6dof-graspnet) [[original repo]](https://github.com/NVlabs/6dof-graspnet) | [paper](https://arxiv.org/abs/1905.10520) |

Install the dependencies of the algorithm you want to benchmark. You can follow the instructions provided by the authors of each algorithm, or you can take a look at the Docker recipes and see how we did it ourselves. The currently supported algorithms are:
    - **Dexnet**:You need to install [gqcnn](https://berkeleyautomation.github.io/gqcnn/)
    - **GPD**: Follow [gpd](https://github.com/atenpas/gpd)
    - **Superquadrics-based grasp planner**: Follow [superquadric-lib](https://github.com/robotology/superquadric-lib). Note that you need to compile the python bindings.
    - **6DoF-GraspNet**: We used [this PyTorch implementation](https://github.com/jsll/pytorch_6dof-graspnet).