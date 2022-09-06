import os
from typing import Tuple

# Importing Typings
import nptyping
from nptyping import Float, Int, NDArray, Shape

from ros_utils import *
from vis_utils import *
# Importing general ROS Info
import rospy

import numpy as np
import yaml

# Importing Grasp Planner ROS Packages
from grasping_benchmarks_ros.srv import (
    GraspPlanner,
    GraspPlannerRequest,
    GraspPlannerResponse,
)

NUMBER_OF_CANDIDATES = 1




def create_grasp_planner_request_from_demo_data(input_folder: str) -> GraspPlannerRequest:
    seg_img = np.load(os.path.join(input_folder, "seg_img.npy"), allow_pickle=True)
    rgb_img = np.load(os.path.join(input_folder, "rgb_img.npy"), allow_pickle=True)
    depth_img = np.load(os.path.join(input_folder, "depth_img.npy"), allow_pickle=True)

    pc = np.load(os.path.join(input_folder, "pointcloud.npz"), allow_pickle=True)
    pc_points = pc["pc_points"]
    pc_colors = pc["pc_colors"]

    with open(os.path.join(input_folder, "cam_info.yaml"), 'r') as stream:
        cam_info = yaml.safe_load(stream)
        print(cam_info.keys())
    cam_pos  = np.array(cam_info["cam_pos"])
    cam_quat = np.array(cam_info["cam_quat"])
    cam_intrinsics = np.array(cam_info["cam_intrinsics"])
    cam_height = cam_info["cam_height"]
    cam_width = cam_info["cam_width"]

    visualize_pointcloud(pc_points, pc_colors, cam_pos)

    return create_grasp_planner_request(
        rgb_img,
        depth_img,
        seg_img,
        pc_points,
        pc_colors,
        cam_pos,
        cam_quat,
        cam_intrinsics,
        cam_height,
        cam_width,
        NUMBER_OF_CANDIDATES
    ), pc_points, pc_colors


def call_grasp_planner(
        input_folder: str, service_id: str
) -> Tuple[NDArray[Shape["3"], Float], NDArray[Shape["4"], Float]]:
    planner_req, pc_points, pc_colors = create_grasp_planner_request_from_demo_data(input_folder)

    rospy.wait_for_service(service_id, timeout=30.0)
    grasp_planner = rospy.ServiceProxy(service_id, GraspPlanner)

    try:
        reply: GraspPlannerResponse = grasp_planner(planner_req)
        print("Service {} reply is: \n{}".format(grasp_planner.resolved_name, reply))

        print([candidate.pose.pose.position.z for candidate in reply.grasp_candidates])
        max_pos_idx = np.argmax([candidate.pose.pose.position.z for candidate in reply.grasp_candidates])
        pose = reply.grasp_candidates[max_pos_idx].pose.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        quat = [
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
        ]
        width = reply.grasp_candidates[max_pos_idx].width.data

        visualize_grasp(pc_points, pc_colors, position, quat, width)
        return position, quat

    except rospy.ServiceException as e:
        print("Service {} call failed: {}".format(grasp_planner.resolved_name, e))


def main():
    # ROS Init
    rospy.init_node("test")


    graspnet_service =      "/graspnet_bench/graspnet_grasp_planner_service"
    superquadrics_service = "/superquadric_bench/superq_grasp_planner_service"
    gpd_service =           "/gpd_bench/gpd_grasp_planner/gpd_grasp_planner_service"
    dexnet_service =        "/dexnet_bench/dexnet_grasp_planner_service"
    # 2. Calling the grasp planner
    pos, quat = call_grasp_planner(
        "example_data",
        service_id=gpd_service
    )


if __name__ == "__main__":
    main()
