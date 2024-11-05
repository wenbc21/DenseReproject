import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm

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

    DRIVE = '2013_05_28_drive_0003_sync'
    seq = "seq_003"
    root_dir = './KITTI_to_colmap/KITTI-colmap'
    data_dir = f'{root_dir}/{DRIVE}/{seq}'
    save_dir = f"colmap_dense_vis/extruded_vis/{DRIVE}_{seq}"
    os.makedirs(save_dir, exist_ok=True)
    
    img_names = sorted(os.listdir(data_dir))
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
    
    pcd = o3d.io.read_point_cloud(f"colmap_dense_vis/extruded_pcd/{DRIVE}_{seq}.ply")
    point_cloud = np.asarray(pcd.points)
    point_color = np.asarray(pcd.colors)
    
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
    depth_max = 50.0
    
    # 增加齐次坐标
    ones = np.ones((point_cloud.shape[0], 1))
    points_world_homogeneous = np.hstack((point_cloud, ones))
    
    for img_ins in tqdm(img_names):
        # 相机外参
        extrinsic = c2w_dict[img_ins]
        extrinsic = np.linalg.inv(extrinsic) #c2w->w2c
        
        # 使用外参矩阵进行变换
        points_camera_homogeneous = extrinsic @ points_world_homogeneous.T

        # 提取深度信息
        depths = points_camera_homogeneous[2, :]

        # 根据深度裁剪点
        mask = (depths > depth_min) & (depths < depth_max)
        points_camera_homogeneous = points_camera_homogeneous[:, mask]
        point_color_l = point_color[mask]
        depths = depths[mask]

        # 进行透视投影
        points_camera = points_camera_homogeneous[:3, :] / points_camera_homogeneous[2, :]

        # 应用内参矩阵
        points_image_homogeneous = K @ points_camera

        # 归一化
        points_image = points_image_homogeneous[:2, :] / points_image_homogeneous[2, :]

        # 对深度进行排序
        sorted_indices = np.argsort(depths)[::-1]
        points_image_sorted = points_image[:, sorted_indices]
        point_color_sorted = point_color_l[sorted_indices]
        
        # rasterize
        image = np.full((H, W, 3), 255, dtype=np.uint8)
        for i in range(points_image_sorted.shape[1]):
            x, y = int(points_image_sorted[0, i]), int(points_image_sorted[1, i])
            if 0 <= x < W and 0 <= y < H:
                color = point_color_sorted[i] * 255
                # cv2.circle(image, (x, y), 2, (int(color[2]), int(color[1]), int(color[0])), -1)
                cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), (int(color[2]), int(color[1]), int(color[0])), -1)

        # 保存图像
        cv2.imwrite(f"{save_dir}/{img_ins}", image)
        