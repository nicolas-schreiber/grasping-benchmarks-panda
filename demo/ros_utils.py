from typing import Optional

import rospy

import numpy as np
import ros_numpy
from geometry_msgs.msg import PoseStamped
from nptyping import Float, Int, NDArray, Shape
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header

from grasping_benchmarks_ros.srv import GraspPlannerRequest


# ========================
#  Value Transformations
# ========================
_FLOAT_EPS = np.finfo(np.float64).eps

def quat2mat(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))

def posRotMat2TFMat(pos, rot_mat):
    """Converts a position and a 3x3 rotation matrix to a 4x4 transformation matrix"""
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

# ==========================
#  Image+PC Transformations
# ==========================
def rgb_float_to_int(rgb_float):
    """
    Covert rgb value from [0, 1] to [0, 255]
    Args:
        rgb_float: rgb array

    Returns:
        int rgb values
    """
    return (rgb_float * 255).astype(dtype=np.uint32)


def rgb_array_to_uint32(rgb_array):
    """
    Pack 3 rgb values into 1 uint32 integer value using binary operations
    Args:
        rgb_array: array [num_samples, num_data=3]

    Returns:
        packed rgb integer value: [num_samples], dtype=np.uint32
        From left to right:
            0-8 bits: place holders (can be extent to additional channel)
            9-16 bits: red
            17-24 bits: green
            25-32 bits: blue
    """
    rgb32 = np.zeros(rgb_array.shape[0], dtype=np.uint32)
    rgb_array = rgb_array.astype(dtype=np.uint32)
    rgb32[:] = (
        np.left_shift(rgb_array[:, 0], 16)
        + np.left_shift(rgb_array[:, 1], 8)
        + rgb_array[:, 2]
    )
    return rgb32

def transform_pc(pc_points, pos, quat, inverse=False):
    mat = posRotMat2TFMat(pos, quat2mat(quat))
    if inverse:
        mat = np.linalg.inv(mat)

    pc_points_extended = np.append(pc_points, np.ones((pc_points.shape[0], 1)), axis=1)

    rotated_pointwise_array = np.einsum("...i, ji->...j", pc_points_extended, mat)

    return rotated_pointwise_array[:, :3]

# ========================
#  MSG Creators
# ========================
def create_grasp_planner_request(
        rgb_img: NDArray[Shape["*, *, 3"], Float],
        depth_img: NDArray[Shape["*, *"], Float],
        seg_img: NDArray[Shape["*, *"], Int],
        pc_points: NDArray[Shape["*, 3"], Float],
        pc_colors: NDArray[Shape["*, 3"], Int],
        cam_pos: NDArray[Shape["3"], Float],
        cam_quat: NDArray[Shape["4"], Float],
        cam_intrinsics: NDArray[Shape["3, 3"], Float],
        cam_height: float,
        cam_width: float,
        num_of_candidates: int
) -> GraspPlannerRequest:
    planner_req = GraspPlannerRequest()

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "rgbd_cam"

    planner_req.color_image = rgb_img_to_ros_msg(rgb_img, header)
    planner_req.depth_image = depth_img_to_ros_msg(depth_img, header)
    planner_req.seg_image = seg_img_to_ros_msg(seg_img, header)
    planner_req.cloud = pc_to_ros_msg(pc_points, pc_colors, header)
    planner_req.view_point = pos_quat_to_ros_msg(cam_pos, cam_quat, header)
    planner_req.camera_info = cam_intrinsics_to_ros_msg(
        cam_intrinsics, cam_height, cam_width, header
    )
    planner_req.n_of_candidates = num_of_candidates

    return planner_req

def pos_quat_to_ros_msg(
    pos: NDArray[Shape["3"], Float],
    quat: NDArray[Shape["4"], Float],
    header: Optional[Header] = None,
) -> PoseStamped:
    pos_msg = PoseStamped()

    if header is not None:
        pos_msg.header = header

    pos_msg.pose.orientation.w = quat[0]
    pos_msg.pose.orientation.x = quat[1]
    pos_msg.pose.orientation.y = quat[2]
    pos_msg.pose.orientation.z = quat[3]
    pos_msg.pose.position.x = pos[0]
    pos_msg.pose.position.y = pos[1]
    pos_msg.pose.position.z = pos[2]

    return pos_msg


def pc_to_ros_msg(
    pc_points: NDArray[Shape["*, 3"], Int],
    pc_colors: NDArray[Shape["*, 3"], Int],
    header: Optional[Header] = None,
) -> PointCloud2:
    pc_data = np.zeros(
        pc_points.shape[0],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.uint32),
        ],
    )
    pc_data["x"] = pc_points[:, 0]
    pc_data["y"] = pc_points[:, 1]
    pc_data["z"] = pc_points[:, 2]

    rgb = rgb_float_to_int(pc_colors)
    pc_data["rgb"] = rgb_array_to_uint32(rgb)

    pc_msg = ros_numpy.msgify(PointCloud2, pc_data)

    if header is not None:
        pc_msg.header = header

    return pc_msg


def depth_img_to_ros_msg(
    depth_img: NDArray[Shape["*, *"], Float], header: Optional[Header] = None
) -> Image:
    depth_img_ui16 = (depth_img * 1000).astype("uint16")
    depth_msg = ros_numpy.msgify(Image, depth_img_ui16, "16UC1")

    if header is not None:
        depth_msg.header = header

    return depth_msg


def rgb_img_to_ros_msg(
    rgb_img: NDArray[Shape["*, *, 3"], Int], header: Optional[Header] = None
) -> Image:
    rgb_msg = ros_numpy.msgify(Image, rgb_img, "rgb8")

    if header is not None:
        rgb_msg.header = header

    return rgb_msg


def seg_img_to_ros_msg(
    seg_img: NDArray[Shape["*, *"], Int], header: Optional[Header] = None
) -> Image:
    seg_msg = ros_numpy.msgify(Image, seg_img.astype(np.uint8), "8UC1")

    if header is not None:
        seg_msg.header = header

    return seg_msg


def cam_intrinsics_to_ros_msg(
    cam_intrinsics: NDArray[Shape["3, 3"], Float],
    height: int,
    width: int,
    header: Optional[Header] = None,
) -> CameraInfo:
    # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    camera_info = CameraInfo()

    if header is not None:
        camera_info.header = header
        # camera_info.cam_frame = header.frame_id

    camera_info.height = height
    camera_info.width = width
    camera_info.distortion_model = "plumb_bob"
    camera_info.D = np.zeros(
        5
    )  # Since we got simulated data all of this is pretty much without distortion
    camera_info.K = cam_intrinsics.flatten()
    camera_info.R = np.eye(
        3, dtype=float
    ).flatten()  # Only in stereo cameras, otherwise diagonal 1 matrix

    P = np.zeros((3, 4))
    P[0:3, 0:3] = cam_intrinsics
    camera_info.P = P.flatten()

    return camera_info


def cam_intrinsics_to_ros(cam):
    # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
    camera_info = CameraInfo()

    camera_info.height = cam.height
    camera_info.width = cam.width

    print(cam.intrinsics.flatten())

    camera_info.distortion_model = "plumb_bob"
    camera_info.D = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0]
    )  # Since we got simulated data all of this is pretty much without distortion
    camera_info.K = cam.intrinsics.flatten()
    camera_info.R = np.array(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )  # Only in stereo cameras, otherwise diagonal 1 matrix
    P = np.zeros((3, 4))
    P[0:3, 0:3] = cam.intrinsics
    camera_info.P = P.flatten()

    return camera_info



