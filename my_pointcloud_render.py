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
    save_dir = f"colmap_dense_vis/{DRIVE}/{seq}/extruded_vis"
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
    
    pcd = o3d.io.read_point_cloud(f"colmap_dense_vis/{DRIVE}/{seq}/extruded_pcd/{DRIVE}_{seq}.ply")
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
    depth_max = 100.0
    
    # 增加齐次坐标
    ones = np.ones((point_cloud.shape[0], 1))
    points_world_homogeneous = np.hstack((point_cloud, ones))
    
    import time
    step1 = 0.0
    step2 = 0.0
    step3 = 0.0
    step4 = 0.0
    step5 = 0.0
    step6 = 0.0
    step7 = 0.0
    step8 = 0.0
    step9 = 0.0
    step10 = 0.0
    
    for img_ins in tqdm(img_names):
        # 相机外参 # 127
        start_time = time.time()
        extrinsic = c2w_dict[img_ins]
        extrinsic = np.linalg.inv(extrinsic) #c2w->w2c
        end_time = time.time()
        step1 += ((end_time - start_time) * 1000)
        start_time = time.time()
        
        # 使用外参矩阵进行变换 # 14736
        points_camera_homogeneous = extrinsic @ points_world_homogeneous.T
        end_time = time.time()
        step2 += ((end_time - start_time) * 1000)
        start_time = time.time()

        # 提取深度信息 # 24
        depths = points_camera_homogeneous[2, :]
        end_time = time.time()
        step3 += ((end_time - start_time) * 1000)
        start_time = time.time()

        # 根据深度裁剪点 # 83715
        mask = (depths > depth_min) & (depths < depth_max)
        points_camera_homogeneous = points_camera_homogeneous[:, mask]
        point_color_l = point_color[mask]
        depths = depths[mask]
        end_time = time.time()
        step4 += ((end_time - start_time) * 1000)
        start_time = time.time()

        # 进行透视投影 # 11922
        points_camera = points_camera_homogeneous[:3, :] / points_camera_homogeneous[2, :]
        end_time = time.time()
        step5 += ((end_time - start_time) * 1000)
        start_time = time.time()

        # 应用内参矩阵 # 4460
        points_image_homogeneous = K @ points_camera
        end_time = time.time()
        step6 += ((end_time - start_time) * 1000)
        start_time = time.time()

        # 归一化 # 3606
        points_image = points_image_homogeneous[:2, :] / points_image_homogeneous[2, :]
        end_time = time.time()
        step7 += ((end_time - start_time) * 1000)
        start_time = time.time()
        
        # 保留视锥体内部的点
        visible_point = (points_image[0, :] >= 0) & (points_image[0, :] < W) & \
             (points_image[1, :] >= 0) & (points_image[1, :] < H)
        points_image = points_image[:, visible_point]
        point_color_l = point_color_l[visible_point] * 255
        depths = depths[visible_point]
        end_time = time.time()
        step8 += ((end_time - start_time) * 1000)
        start_time = time.time()

        # 对深度进行排序 # 114017
        sorted_indices = np.argsort(depths)[::-1]
        points_image_sorted = points_image[:, sorted_indices]
        point_color_sorted = point_color_l[sorted_indices]
        end_time = time.time()
        step9 += ((end_time - start_time) * 1000)
        start_time = time.time()
        
        # rasterize
        image = np.full((H, W, 3), 255, dtype=np.uint8)
        for i in range(points_image_sorted.shape[1]):
            point = points_image_sorted[:, i]
            x, y = int(point[0]), int(point[1])
            color = point_color_sorted[i]
            image[y-2:y+2,x-2:x+2] = (int(color[2]), int(color[1]), int(color[0]))
        end_time = time.time()
        step10 += ((end_time - start_time) * 1000)

        # 保存图像
        cv2.imwrite(f"{save_dir}/{img_ins}", image)
    
    print(step1/1000)
    print(step2/1000)
    print(step3/1000)
    print(step4/1000)
    print(step5/1000)
    print(step6/1000)
    print(step7/1000)
    print(step8/1000)
    print(step9/1000)
    print(step10/1000)