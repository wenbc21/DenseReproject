import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm
import json


if __name__ == '__main__':

    # all the configs are here
    DRIVE = '2013_05_28_drive_0007_sync'
    seq = 'seq_003'
    semantic_pcd_dir = f"colmap_dense_vis/{DRIVE}/{seq}/semantic_pcd"
    os.makedirs(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map", exist_ok=True)
    
    # read point cloud
    pcd = o3d.io.read_point_cloud(os.path.join(semantic_pcd_dir, f"{DRIVE}_{seq}.ply"))
    point_cloud = np.asarray(pcd.points)
    point_color = np.asarray(pcd.colors)
    print(point_cloud.shape, point_color.shape, np.mean(point_cloud[:, 0]),  np.mean(point_cloud[:, 1]), np.mean(point_cloud[:, 2]))
    
    # scaling, otherwise the BEV Map won't be compact
    point_cloud *= 10
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = 1000, 3000 # prior knowledge from KITTI, no normal points outside this (relaxed) range

    # BEV Map initialize for vegetation
    tpd_hf_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.int16)
    btu_hf_vege = (z_max - z_min) * np.ones_like(tpd_hf_vege)
    sem_map_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1), 3), dtype=np.int16)
    
    # BEV Map initialize for other stuff
    tpd_hf_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.int16)
    btu_hf_rest = (z_max - z_min) * np.ones_like(tpd_hf_rest)
    sem_map_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1), 3), dtype=np.int16)
    
    # height sort to take the highest point's semantic as label
    depths = point_cloud[:, 2]
    sorted_indices = np.argsort(depths)
    point_cloud = point_cloud[sorted_indices]
    point_color = point_color[sorted_indices]
    
    # project all points to ground
    for i in tqdm(range(point_cloud.shape[0]), leave=False):
        x, y, z = point_cloud[i]
        r, g, b = point_color[i] * 255
        
        # clip noise
        if z < z_min or z > z_max :
            continue
        _x, _y, _z = int(x - x_min), int(y - y_min), int(z - z_min)
        
        # make BEV Map for vegetation
        if (r, g, b) == (107, 142, 35) :

            if tpd_hf_vege[_y, _x] < _z:
                tpd_hf_vege[_y, _x] = _z
            if btu_hf_vege[_y, _x] > _z:
                btu_hf_vege[_y, _x] = _z
            sem_map_vege[_y, _x] = r, g, b

        # make BEV Map for others
        else :
            if tpd_hf_rest[_y, _x] < _z:
                tpd_hf_rest[_y, _x] = _z
            if btu_hf_rest[_y, _x] > _z:
                btu_hf_rest[_y, _x] = _z
            sem_map_rest[_y, _x] = r, g, b
    
    # save BEV Map
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/semantic_vege.png", sem_map_vege[:, :, ::-1].astype(np.uint16))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/topdown_vege.png", tpd_hf_vege.astype(np.uint16))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/bottomup_vege.png", btu_hf_vege.astype(np.uint16))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/semantic_rest.png", sem_map_rest[:, :, ::-1].astype(np.uint16))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/topdown_rest.png", tpd_hf_rest.astype(np.uint16))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/bottomup_rest.png", btu_hf_rest.astype(np.uint16))
    
    # save BEV Map for visualize and debug (won't be used)
    tpd_hf_vege_vis = tpd_hf_vege / (z_max - z_min) * 255
    btu_hf_vege_vis = btu_hf_vege / (z_max - z_min) * 255
    tpd_hf_rest_vis = tpd_hf_rest / (z_max - z_min) * 255
    btu_hf_rest_vis = btu_hf_rest / (z_max - z_min) * 255
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/semantic_vege_vis.png", sem_map_vege[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/topdown_vege_vis.png", tpd_hf_vege_vis.astype(np.uint8))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/bottomup_vege_vis.png", btu_hf_vege_vis.astype(np.uint8))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/semantic_rest_vis.png", sem_map_rest[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/topdown_rest_vis.png", tpd_hf_rest_vis.astype(np.uint8))
    cv2.imwrite(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/bottomup_rest_vis.png", btu_hf_rest_vis.astype(np.uint8))
    
    # save local position to restore BEV Map to world relative position
    with open(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/position_info.json", "w") as position_info_file:
        json.dump({"x_min":x_min, "y_min":y_min, "z_min":z_min}, position_info_file, indent=4)
    