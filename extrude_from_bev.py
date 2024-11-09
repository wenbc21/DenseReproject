import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm
import json

label_color_dict = {
     7 : (128, 64,128), # 'road'          
     8 : (244, 35,232), # 'sidewalk'      
    11 : ( 70, 70, 70), # 'building'      
    12 : (102,102,156), # 'wall'          
    13 : (190,153,153), # 'fence'         
    17 : (153,153,153), # 'pole'          
    19 : (250,170, 30), # 'traffic light' 
    20 : (220,220,  0), # 'traffic sign'  
    21 : (107,142, 35), # 'vegetation'    
    22 : (152,251,152), # 'terrain'       
    23 : ( 70,130,180), # 'sky'           
    24 : (220, 20, 60), # 'person'        
    25 : (255,  0,  0), # 'rider'         
    26 : (  0,  0,142), # 'car'           
    27 : (  0,  0, 70), # 'truck'         
    28 : (  0, 60,100), # 'bus'           
    31 : (  0, 80,100), # 'train'         
    32 : (  0,  0,230), # 'motorcycle'    
    33 : (119, 11, 32), # 'bicycle'       
    34 : ( 64,128,128), # 'garage'        
    35 : (190,153,153), # 'gate'          
    37 : (153,153,153), # 'smallpole'     
}

def get_points_from_projections(sem_map, tpd_hf, btu_hf, is_vege) :
    points = []
    colors = []
    height = sem_map.shape[0]
    weight = sem_map.shape[1]
    for x in tqdm(range(1, height - 1), desc="Processing"):
        for y in range(1, weight-1) :
            for k in range(btu_hf[x][y], tpd_hf[x][y] + 1) :
                
                sem_val = sem_map[x][y]
                tpd_val = tpd_hf[x][y]
                vege_col = np.array([107,142, 35])
                if k > tpd_val - 1 :
                    points.append([y, x, k])
                    colors.append(sem_val)
                    continue
                if is_vege and k == btu_hf[x][y] :
                    points.append([y, x, k])
                    colors.append(sem_map[x][y])
                    continue
                if np.count_nonzero(tpd_hf[x-1:x+1][y-1:y+1] == tpd_val) != 9 :
                    points.append([y, x, k])
                    colors.append(sem_map[x][y])
                    continue
                if np.sum(np.all(sem_map[x-1:x+1][y-1:y+1] == sem_val, axis=1)) != 9 :
                    points.append([y, x, k])
                    colors.append(sem_map[x][y])
    return np.array(points), np.array(colors)


if __name__ == '__main__':

    # all the configs are here
    DRIVE = '2013_05_28_drive_0007_sync'
    seq = 'seq_003'
    os.makedirs(f"colmap_dense_vis/{DRIVE}/{seq}/extruded_pcd", exist_ok=True)
    
    # read BEV Map
    sem_map_vege = cv2.imread(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/semantic_vege.png", cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    tpd_hf_vege = cv2.imread(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/topdown_vege.png", cv2.IMREAD_UNCHANGED)
    btu_hf_vege = cv2.imread(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/bottomup_vege.png", cv2.IMREAD_UNCHANGED)
    sem_map_rest = cv2.imread(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/semantic_rest.png", cv2.IMREAD_UNCHANGED)[:, :, ::-1]
    tpd_hf_rest = cv2.imread(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/topdown_rest.png", cv2.IMREAD_UNCHANGED)
    btu_hf_rest = cv2.imread(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/bottomup_rest.png", cv2.IMREAD_UNCHANGED)
    
    # get world relation position
    with open(f"colmap_dense_vis/{DRIVE}/{seq}/bev_map/position_info.json", "r") as position_info_file:
        position_info = json.load(position_info_file)
    x_min = position_info["x_min"]
    y_min = position_info["y_min"]
    z_min = position_info["z_min"]
    
    # extrude vegetation points from BEV Map
    points_vege, colors_vege = get_points_from_projections(
        sem_map_vege,
        tpd_hf_vege,
        btu_hf_vege, 
        True
    )
    # extrude other points from BEV Map
    points_rest, colors_rest = get_points_from_projections(
        sem_map_rest,
        tpd_hf_rest,
        btu_hf_rest, 
        False
    )
    points = np.concatenate((points_vege, points_rest), axis=0)
    colors = np.concatenate((colors_vege, colors_rest), axis=0)
    
    # save extruded pcd
    extruded_pcd = o3d.geometry.PointCloud()
    extruded_pcd.points = o3d.utility.Vector3dVector((points + np.array([x_min, y_min, z_min])) / 10)
    extruded_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    o3d.io.write_point_cloud(f"colmap_dense_vis/{DRIVE}/{seq}/extruded_pcd/{DRIVE}_{seq}.ply", extruded_pcd)
