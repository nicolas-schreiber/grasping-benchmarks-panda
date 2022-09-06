import open3d as o3d
import numpy as np


def visualize_pointcloud(points, colors, cam_pos):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.6, origin=[0, 0, 0])
    mesh_frame_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=cam_pos)
    o3d.visualization.draw_geometries([pcd, mesh_frame, mesh_frame_cam])


def visualize_grasp(points, colors, grasp_pos, grasp_quat, width):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

    mesh_frame_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=grasp_pos)
    mesh_frame_grasp.rotate(mesh_frame_grasp.get_rotation_matrix_from_quaternion(grasp_quat))

    gripper = o3d.geometry.OrientedBoundingBox(center=grasp_pos, R=mesh_frame.get_rotation_matrix_from_quaternion(grasp_quat), extent = [0.02, 2*width or 0.1, 0.02])

    o3d.visualization.draw_geometries([pcd, mesh_frame, mesh_frame_grasp, gripper])


