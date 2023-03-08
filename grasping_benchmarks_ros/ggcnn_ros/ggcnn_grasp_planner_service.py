#!/usr/bin/env python3

from pathlib import Path
import logging
from uuid import uuid4
import time
from typing import List
import importlib

import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np

import rospy
import ros_numpy
from geometry_msgs.msg import PoseStamped

from grasping_benchmarks_ros.srv import (
    GraspPlanner,
    GraspPlannerRequest,
    GraspPlannerResponse,
)
from grasping_benchmarks_ros.msg import BenchmarkGrasp

from ggcnn_grasp_planner_pckg.borrowed import visualization as vis
from ggcnn_grasp_planner_pckg.borrowed.export import Exporter
from ggcnn_grasp_planner_pckg.borrowed.datatypes import YCBSimulationDataSample
from ggcnn_grasp_planner_pckg.ggcnn_grasp_planner import ggcnn_get_grasp
from ggcnn_grasp_planner_pckg.utils.dataset_processing.grasp import Grasp6D

mpl.use("Agg")


class ContactGraspNetPlannerService:
    def __init__(
        self,
        service_name: str,
        debug_path: str,
    ):
        logging.info("Starting ContactGraspNetGraspPlannerService")
        self._service = rospy.Service(service_name, GraspPlanner, self.srv_handler)

        self._exporter = Exporter(debug_path)
        logging.info("Service started")

    def _create_sample(self, req: GraspPlannerRequest) -> YCBSimulationDataSample:
        rgb = ros_numpy.numpify(req.color_image)
        depth = (ros_numpy.numpify(req.depth_image) / 1000).astype(np.float32)
        seg = ros_numpy.numpify(req.seg_image)
        camera_matrix = np.array(req.camera_info.K).reshape(3, 3)
        camera_trafo_h = ros_numpy.numpify(
            req.view_point.pose
        )  # 4x4 homogenous tranformation matrix

        sample = YCBSimulationDataSample(
            rgb=rgb,
            depth=depth,
            points=None,
            points_color=None,
            points_segmented=None,
            points_segmented_color=None,
            segmentation=seg,
            cam_intrinsics=camera_matrix,
            cam_pos=camera_trafo_h[:3, 3],
            cam_rot=camera_trafo_h[:3, :3],
            name=f"{time.strftime('%Y%m%d-%H%M%S')}__{uuid4()}",
        )

        return sample

    def _create_response(self, grasps: List[Grasp6D]) -> GraspPlannerResponse:
        response = GraspPlannerResponse()

        # FIXME right no we assume that the grasps are ordered by score but this might be incorrrect
        for i, g in enumerate(grasps):
            grasp_msg = BenchmarkGrasp()

            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()

            pose.pose.position.x = g.position[0]
            pose.pose.position.y = g.position[1]
            pose.pose.position.z = g.position[
                2
            ]  # TODO compare results with negative z-offset (~2cm)

            pose.pose.orientation.w = g.orientation[0]
            pose.pose.orientation.x = g.orientation[1]
            pose.pose.orientation.y = g.orientation[2]
            pose.pose.orientation.z = g.orientation[3]

            grasp_msg.pose = pose

            grasp_msg.score.data = (len(grasps) - i) / len(grasps)
            # grasp_msg.width.data = None

            response.grasp_candidates.append(grasp_msg)

        return response

    def _save_debug_information(
        self, grasp, cam_intrinsics, cam_rot, cam_pos, rgb, name
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        vis.world_grasps_ax(
            ax,
            rgb,
            grasp.position,
            grasp.orientation,
            cam_intrinsics,
            cam_rot,
            cam_pos,
        )
        plt.close(fig)

        export_data = {
            "visualiazation": fig,
            "orientation": grasp.orientation.tolist(),
            "position": grasp.position.tolist(),
        }

        self._exporter(export_data, name)

    def srv_handler(self, req: GraspPlannerRequest) -> GraspPlannerResponse:
        logging.info("Received service call")

        sample = self._create_sample(req)

        grasps = ggcnn_get_grasp(
            sample.depth,
            sample.cam_intrinsics,
            sample.cam_pos,
            R.from_matrix(sample.cam_rot).as_quat()[[3, 0, 1, 2]],
            req.n_of_candidates,
            sample.segmentation,
        )

        logging.info("saving debug information")
        self._save_debug_information(
            grasps[0],
            sample.cam_intrinsics,
            sample.cam_rot,
            sample.cam_pos,
            sample.rgb,
            sample.name,
        )

        response = self._create_response(grasps)
        logging.info("Created response")

        return response


if __name__ == "__main__":
    # the name we give here gets overwritten by the <node name=...> tag from the launch file
    rospy.init_node("ggcnn_graspnet_grasp_planner")

    # relaodinf the logging config is necessary due to ROS logging behavior: https://github.com/ros/ros_comm/issues/1384
    importlib.reload(logging)
    logging.basicConfig(level=logging.INFO)

    ContactGraspNetPlannerService(
        rospy.get_param("~grasp_planner_service_name"),
        Path(rospy.get_param("~debug_path")),
    )

    rospy.spin()
