#!/usr/bin/env python3

from pathlib import Path

import yaml
from matplotlib import pyplot as plt
import ros_numpy

import rospy

from grasping_benchmarks_ros.srv import (
    GraspPlanner,
    GraspPlannerRequest,
    GraspPlannerResponse
)

from grconvnet.preprocessing import Preprocessor
from grconvnet.posprocessing import Postprocessor
from grconvnet.models import GenerativeResnet
from grconvnet.utils.config import module_from_config
from grconvnet.datatypes import CameraData


class GRConvNetGraspPlannerService():
    def __init__(
        self,
        config_file: str,
        service_name: str,
    ):
        # Initialize the ROS service
        self._service = rospy.Service(
            service_name, GraspPlanner, self.srv_handler
        )

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self._preprocessor = module_from_config(config["preprocessor"])
        self._postprocessor = module_from_config(config["postprocessor"])
        self._model = module_from_config(config["model"])
        self._device = config["model"]["device"]

    def srv_handler(self, req: GraspPlannerRequest) -> GraspPlannerResponse:
        rgb = ros_numpy.numpify(req.color_image)
        depth = ros_numpy.numpify(req.depth_image)
        seg = ros_numpy.numpify(req.seg_image)
        pc = ros_numpy.numpify(req.cloud)
        camera_matrix = np.array(req.camera_info.K).reshape(3, 3)
        camera_trafo_h = ros_numpy.numpify(req.view_point.pose) # 4x4 homogenous tranformation matrix

        sample = CameraData(
            rgb = rgb,
            depth = depth,
            points = pc
            segmentation: TensorType[1, "h", "w", torch.uint8]
            name = ""
        )
            cam_intrinsics: NDArray[Shape["3, 3"], Float] = None
            pos_grasps: TensorType["n_pos_grasps", 4, 2] = None
            neg_grasps: TensorType["n_pos_grasps", 4, 2] = None
        
        # n_candidates = req.n_of_candidates if req.n_of_candidates else 1

        # grasps = self.plan_grasp(
        #     camera_data,
        #     n_candidates=n_candidates,
        # )
        # TODO

        response = GraspPlannerResponse()
        for g in grasps:
            response.grasp_candidates.append(g.to_ros_message())

        return response

if __name__ == "__main__":
    # the name we give here gets overwritten by the <node name=...> tag from the launch file
    rospy.init_node("grconvnet_grasp_planner")

    GRConvNetGraspPlannerService.from_config_file(
        Path(rospy.get_param("~config_file")),
        rospy.get_param("~grasp_planner_service_name"),
    )

    rospy.spin()
