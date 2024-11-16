import open3d as o3d
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
import json
from kitti_labels import gaussiancity_label_color_dict, car_palette, building_palette


if __name__ == '__main__':

    # all the configs are here
    parser = argparse.ArgumentParser(description='Merge BEV Maps')
    parser.add_argument('--DRIVE', type = str, default = '2013_05_28_drive_0003_sync')
    parser.add_argument('--save_dir', type = str, default = './results')
    args = parser.parse_args()
    
    DRIVE = args.DRIVE
    sequences = [item.path for item in os.scandir(f"{args.save_dir}/{DRIVE}") if item.is_dir()]
    sequences.sort()
    print(sequences)
    save_dir = f"{args.save_dir}/merged_bev_map/{DRIVE}/bev_map"
    os.makedirs(save_dir, exist_ok=True)
    
    # init merged points
    point_coor_vege = [] # x, y
    point_info_vege = [] # label, btu, tpd
    point_coor_rest = [] # x, y
    point_info_rest = [] # label, btu, tpd
    
    # read BEV Maps
    for sequence in sequences :
        sem_map_vege = cv2.imread(f"{sequence}/bev_map/semantic_vege.png", cv2.IMREAD_UNCHANGED)
        tpd_hf_vege = cv2.imread(f"{sequence}/bev_map/topdown_vege.png", cv2.IMREAD_UNCHANGED)
        btu_hf_vege = cv2.imread(f"{sequence}/bev_map/bottomup_vege.png", cv2.IMREAD_UNCHANGED)
        sem_map_rest = cv2.imread(f"{sequence}/bev_map/semantic_rest.png", cv2.IMREAD_UNCHANGED)
        tpd_hf_rest = cv2.imread(f"{sequence}/bev_map/topdown_rest.png", cv2.IMREAD_UNCHANGED)
        btu_hf_rest = cv2.imread(f"{sequence}/bev_map/bottomup_rest.png", cv2.IMREAD_UNCHANGED)
        
        # get world relation position
        with open(f"{sequence}/bev_map/position_info.json", "r") as position_info_file:
            position_info = json.load(position_info_file)
        x_min = int(position_info["x_min"])
        y_min = int(position_info["y_min"])
        z_min = int(position_info["z_min"])
        
        for x in tqdm(range(sem_map_rest.shape[0]), desc="getting BEV Maps values"):
            for y in range(sem_map_rest.shape[1]) :
                
                sem_val = sem_map_rest[x,y]
                if sem_val != 0 :
                    point_coor_rest.append([x+x_min, y+y_min])
                    point_info_rest.append([sem_val, btu_hf_rest[x,y], tpd_hf_rest[x,y]])
                sem_val = sem_map_vege[x,y]
                if sem_val != 0 :
                    point_coor_vege.append([x+x_min, y+y_min])
                    point_info_vege.append([sem_val, btu_hf_vege[x,y], tpd_hf_vege[x,y]])
    
    point_coor_vege = np.array(point_coor_vege)
    point_info_vege = np.array(point_info_vege)
    point_coor_rest = np.array(point_coor_rest)
    point_info_rest = np.array(point_info_rest)
    
    x_min, x_max = min(np.min(point_coor_vege[:, 0]), np.min(point_coor_rest[:, 0])), max(np.max(point_coor_vege[:, 0]), np.max(point_coor_rest[:, 0]))
    y_min, y_max = min(np.min(point_coor_vege[:, 1]), np.min(point_coor_rest[:, 1])), max(np.max(point_coor_vege[:, 1]), np.max(point_coor_rest[:, 1]))
    z_min, z_max = 1000, 3000 # prior knowledge from KITTI, no normal points outside this (relaxed) range

    # BEV Map initialize for vegetation
    tpd_hf_vege = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    btu_hf_vege = (z_max - z_min) * np.ones_like(tpd_hf_vege)
    sem_map_vege = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    
    # BEV Map initialize for other stuff
    tpd_hf_rest = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    btu_hf_rest = (z_max - z_min) * np.ones_like(tpd_hf_rest)
    sem_map_rest = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1)), dtype=np.int16)
    
    # project vege points to ground
    for i in tqdm(range(point_coor_vege.shape[0]), desc=f"Projecting Vege points to BEV Map"):
        x, y = point_coor_vege[i]
        x -= x_min
        y -= y_min
        label, btu_val, tpd_val = point_info_vege[i]
        
        sem_map_vege[x, y] = label
        tpd_hf_vege[x, y] = tpd_val
        btu_hf_vege[x, y] = btu_val
        
    # project other points to ground
    for i in tqdm(range(point_coor_rest.shape[0]), desc=f"Projecting Other points to BEV Map"):
        x, y = point_coor_rest[i]
        x -= x_min
        y -= y_min
        label, btu_val, tpd_val = point_info_rest[i]
        
        sem_map_rest[x, y] = label
        tpd_hf_rest[x, y] = tpd_val
        btu_hf_rest[x, y] = btu_val
    
    # save BEV Map
    cv2.imwrite(f"{save_dir}/semantic_vege.png", sem_map_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/topdown_vege.png", tpd_hf_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bottomup_vege.png", btu_hf_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/semantic_rest.png", sem_map_rest.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/topdown_rest.png", tpd_hf_rest.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/bottomup_rest.png", btu_hf_rest.astype(np.uint16))
    
    # save BEV Map for visualize and debug (won't be used)
    tpd_hf_vege_vis = tpd_hf_vege / (z_max - z_min) * 255
    btu_hf_vege_vis = btu_hf_vege / (z_max - z_min) * 255
    tpd_hf_rest_vis = tpd_hf_rest / (z_max - z_min) * 255
    btu_hf_rest_vis = btu_hf_rest / (z_max - z_min) * 255
    sem_map_vege_rgb = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1), 3), dtype=np.uint8)
    sem_map_rest_rgb = np.zeros((int(x_max - x_min + 1), int(y_max - y_min + 1), 3), dtype=np.uint8)
    for x in range(sem_map_vege_rgb.shape[0]) :
        for y in range(sem_map_vege_rgb.shape[1]) :
            if sem_map_vege[x,y] != 0 :
                sem_map_vege_rgb[x,y] = gaussiancity_label_color_dict[sem_map_vege[x,y]]
            if sem_map_rest[x,y] != 0 :
                sem_label = sem_map_rest[x,y]
                if sem_label < 100 :
                    sem_map_rest_rgb[x,y] = gaussiancity_label_color_dict[sem_label]
                elif 100 <= sem_label < 10000 :
                    sem_map_rest_rgb[x,y] = car_palette(sem_label) # car
                elif 10000 <= sem_label < 20000 :
                    sem_map_rest_rgb[x,y] = building_palette(sem_label) # building
    cv2.imwrite(f"{save_dir}/semantic_vege_vis.png", sem_map_vege_rgb[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"{save_dir}/topdown_vege_vis.png", tpd_hf_vege_vis.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bottomup_vege_vis.png", btu_hf_vege_vis.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/semantic_rest_vis.png", sem_map_rest_rgb[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"{save_dir}/topdown_rest_vis.png", tpd_hf_rest_vis.astype(np.uint8))
    cv2.imwrite(f"{save_dir}/bottomup_rest_vis.png", btu_hf_rest_vis.astype(np.uint8))
    
    # save local position to restore BEV Map to world relative position
    with open(f"{save_dir}/position_info.json", "w") as position_info_file:
        json.dump({"x_min":int(x_min), "y_min":int(y_min), "z_min":int(z_min)}, position_info_file, indent=4)
