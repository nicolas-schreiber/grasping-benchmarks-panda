#!/usr/bin/env python3

from pathlib import Path

import yaml
from matplotlib import pyplot as plt
import ros_numpy
from scipy.spatial.transform import Rotation

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
from grconvnet.datatypes import YCBData


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
        self._img2world_converter = module_from_config(config["img2world_converter"])

        self._device = config["model"]["device"]

    def srv_handler(self, req: GraspPlannerRequest) -> GraspPlannerResponse:
        rgb = ros_numpy.numpify(req.color_image)
        depth = ros_numpy.numpify(req.depth_image)
        seg = ros_numpy.numpify(req.seg_image)
        pc = ros_numpy.numpify(req.cloud)
        camera_matrix = np.array(req.camera_info.K).reshape(3, 3)
        camera_trafo_h = ros_numpy.numpify(req.view_point.pose) # 4x4 homogenous tranformation matrix
        n_candidates = req.n_of_candidates

        sample = YCBData(
            rgb = rgb,
            depth = depth,
            points = pc,
            segmentation = seg,
            cam_intrinsics = camera_matrix,
            cam_pos = camera_trafo_h[:3, 3],
            cam_rot = camera_trafo_h[:3, :3],
            name = "datapoint from ROS service call",
        )
        
        input_tensor = self._preprocessor(sample)
        input_tensor = input_tensor.to(self._device)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self._model(input_tensor)

        # the number of candidates to return is given in the service request
        # therefore we need to overwrrite the value in the postprocessor
        self._postprocessor.grasp_localizer.grasps = n_candidates

        grasps_img = self._postprocessor(output, sample)

        grasps_world = [
            self._img2world_converter(
                g_img,
                sample.depth,
                sample.cam_intrinsics,
                sample.cam_rot,
                sample.cam_pos
            )
            for g_img in grasps_img
        ]

        # create the response message
        response = GraspPlannerResponse()
        for g in grasps:
            grasp_msg = BenchmarkGrasp()

            pose = PoseStamped()
            # p.header.frame_id = self.ref_frame
            pose.header.stamp = rospy.Time.now()
            
            pose.pose.position.x = g.center.position[0]
            pose.pose.position.y = g.center.position[1]
            pose.pose.position.z = g.center.position[2]

            quat = Rotation.from_euler("z", g.center.angle).as_quat()
            p.pose.orientation.x = self.quaternion[0]
            p.pose.orientation.y = self.quaternion[1]
            p.pose.orientation.z = self.quaternion[2]
            p.pose.orientation.w = self.quaternion[3]
            
            grasp_msg.pose = pose

            grasp_msg.score.data = g.quality
            grasp_msg.width.data = g.width

            response.grasp_candidates.append(grasp_msg)

        return response

if __name__ == "__main__":
    # the name we give here gets overwritten by the <node name=...> tag from the launch file
    rospy.init_node("grconvnet_grasp_planner")

    GRConvNetGraspPlannerService.from_config_file(
        Path(rospy.get_param("~config_file")),
        rospy.get_param("~grasp_planner_service_name"),
    )

    rospy.spin()
