#!/usr/bin/env python3


import yaml

import ros_numpy
import numpy as np
import math

import rospy

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
        service_name: str
    ):
    # Initialize the ROS service 
    # ('service_name' -> ,
    #  'GraspPlanner' -> name of the service type [.srv - file],
    #  'self.service_handler' -> function that takes a service request and returns a service response)
        self._service = rospy.Service(
            service_name, GraspPlanner, self.srv_handler
        )

    def matrix_to_quaternion(self, matrix):
        #Transform rotation from 3x3 matrix to quaternion representation matrix: 3x3
        q = np.empty((4, ), dtype=np.float64)
        M = np.identity(4)
        M[:3, :3] = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2

            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0

            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1

            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]

            q[i] = M[k, i] + M[i, k]
            q[j] = M[i, j] + M[j, i]
            q[k] = t
            q[3] = M[k, j] - M[j, k]

        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q


    def srv_handler(self, request: GraspPlannerRequest) -> GraspPlannerResponse:
        
        
        depth_img = ros_numpy.numpify(request.depth_image)
        depth_img = depth_img/1000

        seg_img = ros_numpy.numpify(request.seg_image)

        camera_matrix = np.array(request.camera_info.K).reshape(3, 3)


        camera_trafo_h = ros_numpy.numpify(request.view_point.pose) # 4x4 homogenous tranformation matrix

        n_candidates = request.n_of_candidates

        cam_intrinsics = camera_matrix
        cam_pos = camera_trafo_h[:3, 3]
        cam_rot = camera_trafo_h[:3, :3]
        cam_quat = self.matrix_to_quaternion(cam_rot)
        #hotfix cam_quat
        
        # insert ggcnn_grasp_planner here
        print("CAM Intrinsics MATRIX that is passed on to grasp planner")
        print(cam_intrinsics)
        grasps6D = ggcnn_get_grasp(depth_img, cam_intrinsics, cam_pos, cam_quat, n_candidates, seg_img = seg_img)

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

            #quat = g.orientation
            pose.pose.orientation.x = g.orientation[0]
            pose.pose.orientation.y = g.orientation[1]
            pose.pose.orientation.z = g.orientation[2]
            pose.pose.orientation.w = g.orientation[3]
            
            grasp_msg.pose = pose

            #grasp_msg.score.data = g.quality
            #grasp_msg.width.data = g.width

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