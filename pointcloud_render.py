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

    # all the configs are here
    DRIVE = '2013_05_28_drive_0003_sync'
    seq = "seq_001"
    root_dir = './KITTI_to_colmap/KITTI-colmap'
    data_dir = f'{root_dir}/{DRIVE}/{seq}'
    save_dir = f"colmap_dense_vis/{DRIVE}/{seq}/extruded_vis"
    os.makedirs(save_dir, exist_ok=True)
    
    # read data
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
    
    # load pose
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
    
    # read point cloud
    pcd = o3d.io.read_point_cloud(f"colmap_dense_vis/{DRIVE}/{seq}/extruded_pcd/{DRIVE}_{seq}.ply")
    point_cloud = np.asarray(pcd.points)
    point_color = np.asarray(pcd.colors)
    
    # intrinsic matrix
    W = 1408
    H = 376
    focal = P_rect_00[0][0]
    cx = P_rect_00[0][2]
    cy = P_rect_00[1][2]
    K = np.array([[focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]])
    
    # depth clip range
    depth_min = 0.1
    depth_max = 100.0
    
    # add homogeneous dimension
    ones = np.ones((point_cloud.shape[0], 1))
    points_world_homogeneous = np.hstack((point_cloud, ones))
    
    # point cloud transform for each camera pose
    for img_ins in tqdm(img_names, desc=f"Rendering camera view"):
        # don't have to render both view
        if img_ins.split('_')[0] == "01" :
            continue
        
        # extrinsic c2w -> w2c
        extrinsic = c2w_dict[img_ins]
        extrinsic = np.linalg.inv(extrinsic)
        
        # extrinsic matrix multiply
        points_camera_homogeneous = extrinsic @ points_world_homogeneous.T

        # depth clip
        depths = points_camera_homogeneous[2, :]
        mask = (depths > depth_min) & (depths < depth_max)
        points_camera_homogeneous = points_camera_homogeneous[:, mask]
        point_color_l = point_color[mask]
        depths = depths[mask]

        # perspective projection
        points_camera = points_camera_homogeneous[:3, :] / points_camera_homogeneous[2, :]

        # intrinsic matrix multiply
        points_image_homogeneous = K @ points_camera

        # normalize
        points_image = points_image_homogeneous[:2, :] / points_image_homogeneous[2, :]
        
        # reserve points in frustrum
        visible_point = (points_image[0, :] >= 0) & (points_image[0, :] < W) & \
             (points_image[1, :] >= 0) & (points_image[1, :] < H)
        points_image = points_image[:, visible_point]
        point_color_l = point_color_l[visible_point] * 255
        depths = depths[visible_point]

        # sort depth
        sorted_indices = np.argsort(depths)[::-1]
        points_image_sorted = points_image[:, sorted_indices]
        point_color_sorted = point_color_l[sorted_indices]
        
        # rasterize 
        # TODO: accelerate this loop
        image = np.full((H, W, 3), 255, dtype=np.uint8)
        for i in range(points_image_sorted.shape[1]):
            point = points_image_sorted[:, i]
            x, y = int(point[0]), int(point[1])
            color = point_color_sorted[i]
            image[y-2:y+2,x-2:x+2] = (int(color[2]), int(color[1]), int(color[0]))

        # save rendered image
        cv2.imwrite(f"{save_dir}/{img_ins}", image)