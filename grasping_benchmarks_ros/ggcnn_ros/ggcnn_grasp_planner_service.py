
import yaml

import ros_numpy
import numpy as np

import rospy
from grasping_benchmarks.base.transformations import matrix_to_quaternion

from grasping_benchmarks_ros.srv import (
    GraspPlanner,
    GraspPlannerRequest,
    GraspPlannerResponse
)

from grasping_benchmarks_ros.msg import BenchmarkGrasp
from geometry_msgs.msg import PoseStamped

from ggcnn_grasp_planner_pckg.ggcnn_grasp_planner import ggcnn_get_grasp


class GGCNNGraspPlannerService():
    def __init__(
        self,
        config_file: str,
        service_name: str,
    ):
    # Initialize the ROS service 
    # ('service_name' -> ,
    #  'GraspPlanner' -> name of the service type [.srv - file],
    #  'self.service_handler' -> function that takes a service request and returns a service response)
        self._service = rospy.Service(
            service_name, GraspPlanner, self.srv_handler
        )

    def srv_handler(self, request: GraspPlannerRequest) -> GraspPlannerResponse:
        
        depth_img = ros_numpy.numpify(request.depth_image)

        camera_matrix = np.array(request.camera_info.K).reshape(3, 3)
        camera_trafo_h = ros_numpy.numpify(request.view_point.pose) # 4x4 homogenous tranformation matrix
        
        n_candidates = request.n_of_candidates

        cam_intrinsics = camera_matrix,
        cam_pos = camera_trafo_h[:3, 3],
        cam_rot = camera_trafo_h[:3, :3],
        cam_quat = matrix_to_quaternion(cam_rot)
        
        # insert ggcnn_grasp_planner here
        grasps6D = ggcnn_get_grasp(depth_img, cam_intrinsics, cam_pos, cam_quat, n_candidates)

        # create the response message
        # response = <service_name>Response 
        # -> format and params are defined in .srv file
        response = GraspPlannerResponse()
        for g in grasps6D:
            # create a message instance, format defined in .msg  file
            grasp_msg = BenchmarkGrasp()

            pose = PoseStamped()
            # p.header.frame_id = self.ref_frame
            pose.header.stamp = rospy.Time.now()
            
            pose.pose.position.x = g.position[0]
            pose.pose.position.y = g.position[1]
            pose.pose.position.z = g.position[2]

            quat = g.orientation
            pose.pose.orientation.x = self.quaternion[0]
            pose.pose.orientation.y = self.quaternion[1]
            pose.pose.orientation.z = self.quaternion[2]
            pose.pose.orientation.w = self.quaternion[3]
            
            grasp_msg.pose = pose

            grasp_msg.score.data = g.quality
            grasp_msg.width.data = g.width

            response.grasp_candidates.append(grasp_msg)

        return response

if __name__ == "__main__":
    # the name we give here gets overwritten by the <node name=...> tag from the launch file
    rospy.init_node("ggcnn_grasp_planner")

    # call class including init function that creates the service
    # gets parameter from launch file
    GGCNNGraspPlannerService(
        
        rospy.get_param("~grasp_planner_service_name"),
    )

    rospy.spin()