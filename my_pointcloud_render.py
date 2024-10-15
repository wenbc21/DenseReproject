import open3d as o3d
from open3d import visualization
import numpy as np
import json
import os
import numpy as np
import shutil
import pyrender
import pyvista as pv
import vtk
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def trans_to_matrix(trans):
    """ Convert a numpy.ndarray to a vtk.vtkMatrix4x4 """
    matrix = vtk.vtkMatrix4x4()
    for i in range(trans.shape[0]):
        for j in range(trans.shape[1]):
            matrix.SetElement(i, j, trans[i, j])
    return matrix

def load_intrinsics(intrinsics_fn):
    with open(intrinsics_fn, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                P_rect_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'P_rect_01:':
                P_rect_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'R_rect_00:':
                R_rect_00 = np.array(line[1:], dtype=np.float32).reshape(3, 3)
            elif line[0] == 'R_rect_01:':
                R_rect_01 = np.array(line[1:], dtype=np.float32).reshape(3, 3)
    return P_rect_00, P_rect_01, R_rect_00, R_rect_01

def load_cam_to_pose(cam_to_pose_fn):
    with open(cam_to_pose_fn, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = line.split(' ')
            if line[0] == 'image_00:':
                c2p_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'image_01:':
                c2p_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
    return c2p_00, c2p_01

if __name__ == '__main__':

    DRIVE = '2013_05_28_drive_0009_sync'
    root_dir = 'KITTI-pre'

    data_dir = f'{root_dir}/data_2d_raw/{DRIVE}'

    start_end_list = [[715,795],[880,960],[1102,1182],[2170,2250],[2900,2980]]

    for idx in range(len(start_end_list)):
        seq_name = f'seq_{idx+1}'
        print(f'Processing sequence: {seq_name}.')
        os.makedirs(f"colmap_dense_vis/{seq_name}", exist_ok=True)
        start = start_end_list[idx][0]
        end = start_end_list[idx][1]
        seq_save_dir = os.path.join(root_dir, seq_name)
        train_save_dir = os.path.join(seq_save_dir, 'train_imgs')
        img_names = sorted(os.listdir(train_save_dir))

        poses_fn = f'{root_dir}/data_poses/{DRIVE}/poses.txt'
        intrinsic_fn = f'{root_dir}/calibration/perspective.txt'
        cam2pose_fn = f'{root_dir}/calibration/calib_cam_to_pose.txt'
        poses = np.loadtxt(poses_fn)
        img_id = poses[:, 0].astype(np.int32)
        poses = poses[:, 1:].reshape(-1, 3, 4)
        pose_dict = {}
        for i in range(len(img_id)):
            img_name = f'{img_id[i]:010d}.png'
            pose_dict[img_name] = poses[i]
        
        P_rect_00, P_rect_01, R_rect_00_, R_rect_01_ = load_intrinsics(intrinsic_fn)
        c2p_00, c2p_01 = load_cam_to_pose(cam2pose_fn)
        c2p_00 = np.concatenate([c2p_00, np.array([[0, 0, 0, 1]])], axis=0)
        c2p_01 = np.concatenate([c2p_01, np.array([[0, 0, 0, 1]])], axis=0)
        R_rect_00 = np.eye(4)
        R_rect_00[:3, :3] = R_rect_00_
        R_rect_01 = np.eye(4)
        R_rect_01[:3, :3] = R_rect_01_
        c2w_dict = {}
        for img_name in pose_dict.keys():
            pose = pose_dict[img_name]
            pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
            c2w_00 = np.matmul(np.matmul(pose, c2p_00), np.linalg.inv(R_rect_00))
            c2w_01 = np.matmul(np.matmul(pose, c2p_01), np.linalg.inv(R_rect_01))
            c2w_dict[f'00_{img_name}'] = c2w_00
            c2w_dict[f'01_{img_name}'] = c2w_01
        

        pcd = o3d.io.read_point_cloud("colmap_res/seq_1/dense/fused.ply")
        point_cloud = np.asarray(pcd.points)
        point_color = np.asarray(pcd.colors)
        
        for img_ins in img_names :
            # 相机外参
            extrinsic = c2w_dict[img_ins]
            extrinsic = np.linalg.inv(extrinsic) #c2w->w2c
            
            # 相机内参
            W = 1408
            H = 376
            focal = P_rect_00[0][0]
            cx = P_rect_00[0][2]
            cy = P_rect_00[1][2]
            K = np.array([[focal, 0, cx],
                        [0, focal, cy],
                        [0, 0, 1]])
            
            # 深度裁剪范围
            depth_min = 0.1
            depth_max = 100.0
            
            # 增加齐次坐标
            ones = np.ones((point_cloud.shape[0], 1))
            points_world_homogeneous = np.hstack((point_cloud, ones))
            
            # 使用外参矩阵进行变换
            points_camera_homogeneous = extrinsic @ points_world_homogeneous.T

            # 提取深度信息
            depths = points_camera_homogeneous[2, :]

            # 根据深度裁剪点
            mask = (depths > depth_min) & (depths < depth_max)
            points_camera_homogeneous = points_camera_homogeneous[:, mask]
            point_color_l = point_color[mask]

            # 进行透视投影
            points_camera = points_camera_homogeneous[:3, :] / points_camera_homogeneous[2, :]

            # 应用内参矩阵
            points_image_homogeneous = K @ points_camera

            # 归一化
            points_image = points_image_homogeneous[:2, :] / points_image_homogeneous[2, :]
            
            # rasterize
            image = np.full((H, W, 3), 255, dtype=np.uint8)
            for i in range(points_image.shape[1]):
                x, y = int(points_image[0, i]), int(points_image[1, i])
                color = point_color_l[i] * 255
                cv2.circle(image, (x, y), 2, (int(color[2]), int(color[1]), int(color[0])), -1)  # 绘制圆点，半径为5

            # 保存图像
            cv2.imwrite(f"colmap_dense_vis/{seq_name}/{img_ins}_img.png", image)
            
            # # renderer
            # p = pv.Plotter(off_screen=True, window_size=[W,H])

            # #
            # # load mesh or point cloud
            # #
            # mesh = pv.read(f"colmap_res/{seq_name}/dense/fused.ply")
            # p.add_mesh(mesh, rgb=True)

            # # convert the principal point to window center (normalized coordinate system) and set it
            # wcx = -2*(cx - float(W)/2) / W
            # wcy =  2*(cy - float(H)/2) / H
            # p.camera.SetWindowCenter(wcx, wcy)

            # # convert the focal length to view angle and set it
            # view_angle = 180 / math.pi * (2.0 * math.atan2(H/2.0, focal))
            # p.camera.SetViewAngle(view_angle)

            # # apply the transform to scene objects
            # p.camera.SetModelTransformMatrix(trans_to_matrix(extrinsic))

            # # the camera can stay at the origin because we are transforming the scene objects
            # p.camera.SetPosition(0, 0, 0)

            # # look in the +Z direction of the camera coordinate system
            # p.camera.SetFocalPoint(0, 0, 1)

            # # the camera Y axis points down
            # p.camera.SetViewUp(0,-1,0)


            # #
            # # near/far plane
            # #

            # # ensure the relevant range of depths are rendered
            # depth_min = 0.01
            # depth_max = 100
            # p.camera.SetClippingRange(depth_min, depth_max)
            # # depth_min, depth_max = p.camera.GetClippingRange()
            # p.renderer.ResetCameraClippingRange()

            # p.show()
            # p.render()
            # p.store_image = True  # last_image and last_image_depth
            # p.close()


            # # get screen image
            # img = p.last_image

            # # get depth
            # # img_depth = p.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False)

            # img = img.astype(np.uint8)  # 转换为 uint8 类型

            # # 将 NumPy 数组转换为图片
            # img = Image.fromarray(img)

            # # 保存图片
            # img.save(f"colmap_dense_vis/{seq_name}/{img_ins}_img.png")
            