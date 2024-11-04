import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm

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

    DRIVE = '2013_05_28_drive_0003_sync'
    semantic_pcd_dir = f"colmap_dense_vis/semantic_pcd"

    os.makedirs(f"colmap_dense_vis/bev_map", exist_ok=True)
    os.makedirs(f"colmap_dense_vis/extruded_pcd", exist_ok=True)
    
    pcd = o3d.io.read_point_cloud(os.path.join(semantic_pcd_dir, f"{DRIVE}.ply"))
    point_cloud = np.asarray(pcd.points)
    point_color = np.asarray(pcd.colors)
    print(point_cloud.shape, point_color.shape)
    
    point_cloud *= 10
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

    tpd_hf_rest = np.zeros((int(y_max - y_min + 2), int(x_max - x_min + 2)), dtype=np.int16)
    btu_hf_rest = z_max * np.ones_like(tpd_hf_rest)
    sem_map_rest = np.zeros((int(y_max - y_min + 2), int(x_max - x_min + 2), 3), dtype=np.int16)
    
    tpd_hf_vege = np.zeros((int(y_max - y_min + 2), int(x_max - x_min + 2)), dtype=np.int16)
    btu_hf_vege = z_max * np.ones_like(tpd_hf_vege)
    sem_map_vege = np.zeros((int(y_max - y_min + 2), int(x_max - x_min + 2), 3), dtype=np.int16)
    
    depths = point_cloud[:, 2]
    sorted_indices = np.argsort(depths)
    point_cloud = point_cloud[sorted_indices]
    point_color = point_color[sorted_indices]
    
    for i in tqdm(range(point_cloud.shape[0]), leave=False):
        x, y, z = point_cloud[i]
        r, g, b = point_color[i] * 255
        _x, _y, _z = int(x - x_min), int(y - y_min), int(z - z_min)
        
        if (r, g, b) == (107, 142, 35) :

            if tpd_hf_vege[_y, _x] < _z:
                tpd_hf_vege[_y, _x] = _z
            if btu_hf_vege[_y, _x] > _z:
                btu_hf_vege[_y, _x] = _z
            sem_map_vege[_y, _x] = r, g, b

        else :
            if tpd_hf_rest[_y, _x] < _z:
                tpd_hf_rest[_y, _x] = _z
            if btu_hf_rest[_y, _x] > _z:
                btu_hf_rest[_y, _x] = _z
            sem_map_rest[_y, _x] = r, g, b
    
    cv2.imwrite(f"colmap_dense_vis/bev_map/{DRIVE}_semantic_vege.png", sem_map_vege[:, ::-1])
    cv2.imwrite(f"colmap_dense_vis/bev_map/{DRIVE}_topdown_vege.png", tpd_hf_vege)
    cv2.imwrite(f"colmap_dense_vis/bev_map/{DRIVE}_bottomup_vege.png", btu_hf_vege)
    cv2.imwrite(f"colmap_dense_vis/bev_map/{DRIVE}_semantic_rest.png", sem_map_rest[:, ::-1])
    cv2.imwrite(f"colmap_dense_vis/bev_map/{DRIVE}_topdown_rest.png", tpd_hf_rest)
    cv2.imwrite(f"colmap_dense_vis/bev_map/{DRIVE}_bottomup_rest.png", btu_hf_rest)
    
    points_vege, colors_vege = get_points_from_projections(
        sem_map_vege,
        tpd_hf_vege.astype(int),
        btu_hf_vege.astype(int), 
        True
    )
    points_rest, colors_rest = get_points_from_projections(
        sem_map_rest,
        tpd_hf_rest.astype(int),
        btu_hf_rest.astype(int), 
        False
    )
    points = np.concatenate((points_vege, points_rest), axis=0)
    colors = np.concatenate((colors_vege, colors_rest), axis=0)
    
    extruded_pcd = o3d.geometry.PointCloud()
    extruded_pcd.points = o3d.utility.Vector3dVector((points + np.array([x_min, y_min, z_min])) / 10)
    extruded_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    o3d.io.write_point_cloud(f"colmap_dense_vis/extruded_pcd/{DRIVE}.ply", extruded_pcd)

