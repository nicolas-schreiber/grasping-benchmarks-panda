#!/usr/bin/env python3

from pathlib import Path

import yaml
from matplotlib import pyplot as plt

import rospy

from grasping_benchmarks_ros.srv import (
    GraspPlanner,
    GraspPlannerRequest,
    GraspPlannerResponse,
    GraspVisualizer,
    GraspVisualizerRequest,
    GraspVisualizerResponse,
)

# from grasping_benchmarks.base import CameraData

# from grasping_benchmarks.grconvnet.grconvnet_grasp_planner import GRConvNetGraspPlanner


# class GRConvNetGraspPlannerService(GRConvNetGraspPlanner):
#     def __init__(
#         self,
#         grasp_service_name: str,
#         grasp_planner_topic_name: str,
#         visualization_service_name: str,
#         *args,
#         **kwargs
#     ):
#         super().__init__(*args, **kwargs)

#         # Initialize the ROS service
#         self._grasp_planning_service = rospy.Service(
#             grasp_service_name, GraspPlanner, self.plan_grasp_srv_handler
#         )

#         self._visualization_service = rospy.Service(
#             visualization_service_name, GraspVisualizer, self.visualize_srv_handler
#         )

#         # TODO: implement publisher

#     def plan_grasp_srv_handler(self, req: GraspPlannerRequest) -> GraspPlannerResponse:
#         camera_data = CameraData.from_grasp_planner_request(req)

#         n_candidates = req.n_of_candidates if req.n_of_candidates else 1

#         grasps = self.plan_grasp(
#             camera_data,
#             n_candidates=n_candidates,
#         )

#         response = GraspPlannerResponse()
#         for g in grasps:
#             response.grasp_candidates.append(g.to_ros_message())

#         return response

#     def visualize_srv_handler(
#         self, req: GraspVisualizerRequest
#     ) -> GraspVisualizerResponse:
#         target_path = Path(req.target_path)

#         fig = self.visualize()
#         fig.savefig(target_path)

#         response = GraspVisualizerResponse()
#         return response


# if __name__ == "__main__":
#     rospy.init_node("grconvnet_grasp_planner")

#     # TODO make parameters from the config gile rosparameters

#     GRConvNetGraspPlannerService.from_config_file(
#         Path(rospy.get_param("~config_file")),
#         rospy.get_param("~grasp_planner_service_name"),
#         rospy.get_param("~grasp_planner_topic_name"),
#         rospy.get_param("~visualization_service_name"),
#     )

#     rospy.spin()
