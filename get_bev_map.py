import open3d as o3d
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
from collections import Counter
import json
from kitti_labels import gaussiancity_label_color_dict, car_palette, building_palette


# something like Label Propagation
def semantic_propagation_denoise(sem_map, tpd_hf, btu_hf, is_vege) :

    height = sem_map.shape[0]
    weight = sem_map.shape[1]
    window_size = 5
    half_window = 2
    tpd_basis = 0
    btu_basis = 3000 - 1000 # prior knowledge from KITTI
    
    for x in tqdm(range(half_window, height - half_window), desc=f"Denoising for BEV Map"):
        for y in range(half_window, weight - half_window) :
            
            sem_val = sem_map[x,y]
            if sem_val == 0 :
                continue
            
            # local window
            window = sem_map[x-half_window:x+half_window+1, y-half_window:y+half_window+1]
            window_flatten = window.flatten()
            window_flatten = window_flatten[window_flatten != 0]
            
            # simply remove for vege points
            if is_vege and window_flatten.size < 10 :
                sem_map[x,y] = 0
                tpd_hf[x,y] = tpd_basis
                btu_hf[x,y] = btu_basis
                continue
            
            # replace or remove for others
            if np.count_nonzero(window_flatten == sem_val) < 5 :
                
                value_counts = Counter(window_flatten)
                most_common_value, most_common_count = value_counts.most_common(1)[0]
                
                # if some other semantic dominating, replace
                if most_common_count >= 3 :
                    most_common_indices = np.argwhere(window == most_common_value)
                    tpd_window = tpd_hf[x-half_window:x+half_window+1, y-half_window:y+half_window+1]
                    btu_window = btu_hf[x-half_window:x+half_window+1, y-half_window:y+half_window+1]
                    average_tpd = np.mean(tpd_window[most_common_indices[:, 0], most_common_indices[:, 1]])
                    average_btu = np.mean(btu_window[most_common_indices[:, 0], most_common_indices[:, 1]])
                    sem_map[x,y] = most_common_value
                    tpd_hf[x,y] = average_tpd
                    btu_hf[x,y] = average_btu

                # if not, remove
                else :
                    sem_map[x,y] = 0
                    tpd_hf[x,y] = tpd_basis
                    btu_hf[x,y] = btu_basis
    
    return sem_map, tpd_hf, btu_hf


if __name__ == '__main__':

    # all the configs are here
    parser = argparse.ArgumentParser(description='Get BEV Map')
    parser.add_argument('--DRIVE', type = str, default = '2013_05_28_drive_0003_sync')
    parser.add_argument('--seq', type = str, default = 'seq_001')
    parser.add_argument('--save_dir', type = str, default = './results')
    args = parser.parse_args()
    
    DRIVE = args.DRIVE
    seq = args.seq
    save_dir = f"{args.save_dir}/{DRIVE}/{seq}"
    semantic_pcd_dir = f"{save_dir}/semantic_pcd"
    os.makedirs(f"{save_dir}/bev_map", exist_ok=True)
    
    # read point cloud
    point_cloud = []
    point_label = []
    with open(os.path.join(semantic_pcd_dir, f"{DRIVE}_{seq}.txt"), "r") as semantic_pcd_txt:
        for line in semantic_pcd_txt:
            data = line.strip().split()
            x, y, z, label = float(data[0]), float(data[1]), float(data[2]), int(data[3])
            point_cloud.append([x, y, z])
            point_label.append(label)
    point_cloud = np.array(point_cloud)
    point_label = np.array(point_label)
    print(point_cloud.shape, point_label.shape, np.mean(point_cloud[:, 0]), np.mean(point_cloud[:, 1]), np.mean(point_cloud[:, 2]))
    
    # scaling, otherwise the BEV Map won't be compact
    point_cloud *= 10
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = 1000, 3000 # prior knowledge from KITTI, no normal points outside this (relaxed) range

    # BEV Map initialize for vegetation
    tpd_hf_vege = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    btu_hf_vege = (z_max - z_min) * np.ones_like(tpd_hf_vege)
    sem_map_vege = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    
    # BEV Map initialize for other stuff
    tpd_hf_rest = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    btu_hf_rest = (z_max - z_min) * np.ones_like(tpd_hf_rest)
    sem_map_rest = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    
    # height sort to take the highest point's semantic as label
    depths = point_cloud[:, 2]
    sorted_indices = np.argsort(depths)
    point_cloud = point_cloud[sorted_indices]
    point_label = point_label[sorted_indices]
    
    # project all points to ground
    for i in tqdm(range(point_cloud.shape[0]), desc=f"Projecting points to BEV Map"):
        x, y, z = point_cloud[i]
        semantic_label = point_label[i]
        
        # clip noise
        if z < z_min or z > z_max :
            continue
        _x, _y, _z = int(x - x_min), int(y - y_min), int(z - z_min)
        
        # make BEV Map for vegetation
        if semantic_label == 21 :
            if tpd_hf_vege[_x, _y] < _z:
                tpd_hf_vege[_x, _y] = _z
            if btu_hf_vege[_x, _y] > _z:
                btu_hf_vege[_x, _y] = _z
            sem_map_vege[_x, _y] = semantic_label

        # make BEV Map for others
        else :
            if tpd_hf_rest[_x, _y] < _z:
                tpd_hf_rest[_x, _y] = _z
            if btu_hf_rest[_x, _y] > _z:
                btu_hf_rest[_x, _y] = _z
            sem_map_rest[_x, _y] = semantic_label
    
    # denoise
    sem_map_vege, tpd_hf_vege, btu_hf_vege = semantic_propagation_denoise(sem_map_vege, tpd_hf_vege, btu_hf_vege, True)
    sem_map_rest, tpd_hf_rest, btu_hf_rest = semantic_propagation_denoise(sem_map_rest, tpd_hf_rest, btu_hf_rest, False)
    
    # semantic merge
    for x in tqdm(range(sem_map_rest.shape[0]), desc=f"Semantic merge for BEV Map"):
        for y in range(sem_map_rest.shape[1]) :
            # merge ground to sidewalk
            if sem_map_rest[x, y] == 6 :
                sem_map_rest[x, y] = 8
            # merge terrain to vegetation
            if sem_map_rest[x, y] == 22 :
                sem_map_rest[x, y] = 21
    
    # save BEV Map
    cv2.imwrite(f"{save_dir}/bev_map/semantic_vege.png", sem_map_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bev_map/topdown_vege.png", tpd_hf_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bev_map/bottomup_vege.png", btu_hf_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bev_map/semantic_rest.png", sem_map_rest.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bev_map/topdown_rest.png", tpd_hf_rest.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bev_map/bottomup_rest.png", btu_hf_rest.astype(np.uint16))
    
    # save BEV Map for visualize and debug (won't be used)
    tpd_hf_vege_vis = tpd_hf_vege / (z_max - z_min) * 255
    btu_hf_vege_vis = btu_hf_vege / (z_max - z_min) * 255
    tpd_hf_rest_vis = tpd_hf_rest / (z_max - z_min) * 255
    btu_hf_rest_vis = btu_hf_rest / (z_max - z_min) * 255
    sem_map_vege_rgb = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1), 3), dtype=np.uint8)
    sem_map_rest_rgb = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1), 3), dtype=np.uint8)
    for x in range(sem_map_vege_rgb.shape[0]) :
        for y in range(sem_map_vege_rgb.shape[1]) :
            if sem_map_vege[x, y] != 0 :
                sem_map_vege_rgb[x, y] = gaussiancity_label_color_dict[sem_map_vege[x, y]]
            if sem_map_rest[x, y] != 0 :
                sem_label = sem_map_rest[x, y]
                if sem_label < 100 :
                    sem_map_rest_rgb[x, y] = gaussiancity_label_color_dict[sem_label]
                elif 100 <= sem_label < 10000 :
                    sem_map_rest_rgb[x, y] = car_palette(sem_label) # car
                elif 10000 <= sem_label < 20000 :
                    sem_map_rest_rgb[x, y] = building_palette(sem_label) # building
    cv2.imwrite(f"{save_dir}/bev_map/semantic_vege_vis.png", sem_map_vege_rgb[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bev_map/topdown_vege_vis.png", tpd_hf_vege_vis.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bev_map/bottomup_vege_vis.png", btu_hf_vege_vis.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bev_map/semantic_rest_vis.png", sem_map_rest_rgb[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bev_map/topdown_rest_vis.png", tpd_hf_rest_vis.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bev_map/bottomup_rest_vis.png", btu_hf_rest_vis.astype(np.uint8))
    
    # save local position to restore BEV Map to world relative position
    with open(f"{save_dir}/bev_map/position_info.json", "w") as position_info_file:
        json.dump({"x_min":x_min, "y_min":y_min, "z_min":z_min}, position_info_file, indent=4)
    