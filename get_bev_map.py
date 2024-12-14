import open3d as o3d
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
from collections import Counter
import json
from kitti_labels import label_color_dict, semantic_classes_dict, class_color_dict, car_palette, building_palette


def _get_point_map(map_size, stride):
    pts_map = np.zeros(map_size, dtype=np.uint8)
    ys = np.arange(0, map_size[0], stride)
    xs = np.arange(0, map_size[1], stride)
    coords = np.stack(np.meshgrid(ys, xs), axis=-1).reshape(-1, 2)
    pts_map[coords[:, 0], coords[:, 1]] = 1
    return pts_map

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
    tpd_hf_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    btu_hf_vege = (z_max - z_min) * np.ones_like(tpd_hf_vege)
    sem_map_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    
    # BEV Map initialize for other stuff
    tpd_hf_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    btu_hf_rest = (z_max - z_min) * np.ones_like(tpd_hf_rest)
    sem_map_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    
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
            if tpd_hf_vege[_y, _x] < _z:
                tpd_hf_vege[_y, _x] = _z
            if btu_hf_vege[_y, _x] > _z:
                btu_hf_vege[_y, _x] = _z
            sem_map_vege[_y, _x] = semantic_label

        # make BEV Map for others
        else :
            if tpd_hf_rest[_y, _x] < _z:
                tpd_hf_rest[_y, _x] = _z
            if btu_hf_rest[_y, _x] > _z:
                btu_hf_rest[_y, _x] = _z
            sem_map_rest[_y, _x] = semantic_label
    
    # denoise
    sem_map_vege, tpd_hf_vege, btu_hf_vege = semantic_propagation_denoise(sem_map_vege, tpd_hf_vege, btu_hf_vege, True)
    sem_map_rest, tpd_hf_rest, btu_hf_rest = semantic_propagation_denoise(sem_map_rest, tpd_hf_rest, btu_hf_rest, False)
    
    # semantic to classes
    for sem_type in semantic_classes_dict :
        sem_map_rest[sem_map_rest == sem_type] = semantic_classes_dict[sem_type]
        sem_map_vege[sem_map_vege == sem_type] = semantic_classes_dict[sem_type]
    
    # classes to color
    sem_map_vege_rgb = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1), 3), dtype=np.uint8)
    sem_map_rest_rgb = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1), 3), dtype=np.uint8)
    
    for x in range(sem_map_vege_rgb.shape[0]) :
        for y in range(sem_map_vege_rgb.shape[1]) :
            if sem_map_vege[x, y] != 0 :
                sem_map_vege_rgb[x, y] = class_color_dict[sem_map_vege[x, y]]
            if sem_map_rest[x, y] != 0 :
                sem_class = sem_map_rest[x, y]
                if sem_class < 100 :
                    sem_map_rest_rgb[x, y] = class_color_dict[sem_class]
                elif 100 <= sem_class < 10000 :
                    sem_map_rest_rgb[x, y] = car_palette(sem_class) # car
                elif 10000 <= sem_class < 20000 :
                    sem_map_rest_rgb[x, y] = building_palette(sem_class) # building
    
    # classes to scale
    pts_map_vege = np.zeros(sem_map_vege.shape, dtype=np.uint8)
    pts_map_vege[sem_map_vege == 3] = 1
    
    pts_map_rest = np.zeros(sem_map_rest.shape, dtype=np.uint8)
    pts_map_rest[sem_map_rest != 0] = 1
    for lab_2 in [1, 5] :
        mask = sem_map_rest == lab_2
        pts_map_rest[mask] = 0
        pt_map = _get_point_map(pts_map_rest.shape, 2)
        pt_map[~mask] = 0
        pts_map_rest += pt_map
    
    # save BEV Map
    print("saving BEV Maps...")
    cv2.imwrite(f"{save_dir}/bev_map/VEGT-PTS.png", pts_map_vege * 255)
    cv2.imwrite(f"{save_dir}/bev_map/VEGT-INS.png", sem_map_vege)
    cv2.imwrite(f"{save_dir}/bev_map/VEGT-SEG.png", sem_map_vege_rgb[:, :, ::-1])
    cv2.imwrite(f"{save_dir}/bev_map/VEGT-TD_HF.png", tpd_hf_vege)
    cv2.imwrite(f"{save_dir}/bev_map/VEGT-BU_HF.png", btu_hf_vege)
    cv2.imwrite(f"{save_dir}/bev_map/REST-PTS.png", pts_map_rest * 255)
    cv2.imwrite(f"{save_dir}/bev_map/REST-INS.png", sem_map_rest)
    cv2.imwrite(f"{save_dir}/bev_map/REST-SEG.png", sem_map_rest_rgb[:, :, ::-1])
    cv2.imwrite(f"{save_dir}/bev_map/REST-TD_HF.png", tpd_hf_rest)
    cv2.imwrite(f"{save_dir}/bev_map/REST-BU_HF.png", btu_hf_rest)
    
    # save local position to restore BEV Map to world relative position
    with open(f"{save_dir}/bev_map/metadata.json", "w") as metadata_file:
        json.dump({"bounds":{"xmin":x_min, "ymin":y_min, "zmin":z_min}}, metadata_file, indent=4)
