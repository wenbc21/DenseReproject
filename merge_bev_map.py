import open3d as o3d
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
import json
from kitti_labels import label_color_dict, class_color_dict, car_palette, building_palette


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
        pts_map_vege = cv2.imread(f"{sequence}/bev_map/VEGT-PTS.png", cv2.IMREAD_UNCHANGED)
        ins_map_vege = cv2.imread(f"{sequence}/bev_map/VEGT-INS.png", cv2.IMREAD_UNCHANGED)
        tpd_hf_vege = cv2.imread(f"{sequence}/bev_map/VEGT-TD_HF.png", cv2.IMREAD_UNCHANGED)
        btu_hf_vege = cv2.imread(f"{sequence}/bev_map/VEGT-BU_HF.png", cv2.IMREAD_UNCHANGED)
        pts_map_rest = cv2.imread(f"{sequence}/bev_map/REST-PTS.png", cv2.IMREAD_UNCHANGED)
        ins_map_rest = cv2.imread(f"{sequence}/bev_map/REST-INS.png", cv2.IMREAD_UNCHANGED)
        tpd_hf_rest = cv2.imread(f"{sequence}/bev_map/REST-TD_HF.png", cv2.IMREAD_UNCHANGED)
        btu_hf_rest = cv2.imread(f"{sequence}/bev_map/REST-BU_HF.png", cv2.IMREAD_UNCHANGED)
        
        # get world relation position
        with open(f"{sequence}/bev_map/metadata.json", "r") as metadata_file:
            metadata = json.load(metadata_file)
        x_min = int(metadata["bounds"]["xmin"])
        y_min = int(metadata["bounds"]["ymin"])
        z_min = int(metadata["bounds"]["zmin"])
        
        for y in tqdm(range(ins_map_rest.shape[0]), desc="getting BEV Maps values"):
            for x in range(ins_map_rest.shape[1]) :
                
                ins_val = ins_map_rest[y,x]
                if ins_val != 0 :
                    point_coor_rest.append([x+x_min, y+y_min])
                    point_info_rest.append([ins_val, btu_hf_rest[y,x], tpd_hf_rest[y,x], pts_map_rest[y,x]])
                ins_val = ins_map_vege[y,x]
                if ins_val != 0 :
                    point_coor_vege.append([x+x_min, y+y_min])
                    point_info_vege.append([ins_val, btu_hf_vege[y,x], tpd_hf_vege[y,x], pts_map_vege[y,x]])
    
    point_coor_vege = np.array(point_coor_vege)
    point_info_vege = np.array(point_info_vege)
    point_coor_rest = np.array(point_coor_rest)
    point_info_rest = np.array(point_info_rest)
    
    x_min, x_max = min(np.min(point_coor_vege[:, 0]), np.min(point_coor_rest[:, 0])), max(np.max(point_coor_vege[:, 0]), np.max(point_coor_rest[:, 0]))
    y_min, y_max = min(np.min(point_coor_vege[:, 1]), np.min(point_coor_rest[:, 1])), max(np.max(point_coor_vege[:, 1]), np.max(point_coor_rest[:, 1]))
    z_min, z_max = 1000, 3000 # prior knowledge from KITTI, no normal points outside this (relaxed) range

    # BEV Map initialize for vegetation
    tpd_hf_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    btu_hf_vege = (z_max - z_min) * np.ones_like(tpd_hf_vege)
    sem_map_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    pts_map_vege = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint8)
    
    # BEV Map initialize for other stuff
    tpd_hf_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    btu_hf_rest = (z_max - z_min) * np.ones_like(tpd_hf_rest)
    sem_map_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint16)
    pts_map_rest = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1)), dtype=np.uint8)
    
    # project vege points to ground
    for i in tqdm(range(point_coor_vege.shape[0]), desc=f"Projecting Vege points to BEV Map"):
        x, y = point_coor_vege[i]
        x -= x_min
        y -= y_min
        label, btu_val, tpd_val, pts_val = point_info_vege[i]
        
        sem_map_vege[y, x] = label
        tpd_hf_vege[y, x] = tpd_val
        btu_hf_vege[y, x] = btu_val
        pts_map_vege[y, x] = pts_val
        
    # project other points to ground
    for i in tqdm(range(point_coor_rest.shape[0]), desc=f"Projecting Other points to BEV Map"):
        x, y = point_coor_rest[i]
        x -= x_min
        y -= y_min
        label, btu_val, tpd_val, pts_val = point_info_rest[i]
        
        sem_map_rest[y, x] = label
        tpd_hf_rest[y, x] = tpd_val
        btu_hf_rest[y, x] = btu_val
        pts_map_rest[y, x] = pts_val
    
    sem_map_vege_rgb = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1), 3), dtype=np.uint8)
    sem_map_rest_rgb = np.zeros((int(y_max - y_min + 1), int(x_max - x_min + 1), 3), dtype=np.uint8)
    
    for x in range(sem_map_vege_rgb.shape[0]) :
        for y in range(sem_map_vege_rgb.shape[1]) :
            if sem_map_vege[x,y] != 0 :
                sem_map_vege_rgb[x,y] = class_color_dict[sem_map_vege[x,y]]
            if sem_map_rest[x,y] != 0 :
                sem_class = sem_map_rest[x,y]
                if sem_class < 100 :
                    sem_map_rest_rgb[x,y] = class_color_dict[sem_class]
                elif 100 <= sem_class < 10000 :
                    sem_map_rest_rgb[x,y] = car_palette(sem_class) # car
                elif 10000 <= sem_class < 20000 :
                    sem_map_rest_rgb[x,y] = building_palette(sem_class) # building
    
    # save BEV Map
    print("saving BEV Maps...")
    cv2.imwrite(f"{save_dir}/VEGT-SEG.png", sem_map_vege_rgb[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"{save_dir}/VEGT-PTS.png", pts_map_vege)
    cv2.imwrite(f"{save_dir}/VEGT-INS.png", sem_map_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/VEGT-TD_HF.png", tpd_hf_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/VEGT-BU_HF.png", btu_hf_vege.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/REST-SEG.png", sem_map_rest_rgb[:, :, ::-1].astype(np.uint8))
    cv2.imwrite(f"{save_dir}/REST-PTS.png", pts_map_rest)
    cv2.imwrite(f"{save_dir}/REST-INS.png", sem_map_rest.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/REST-TD_HF.png", tpd_hf_rest.astype(np.uint16))
    cv2.imwrite(f"{save_dir}/REST-BU_HF.png", btu_hf_rest.astype(np.uint16))
    
    # save local position to restore BEV Map to world relative position
    with open(f"{save_dir}/metadata.json", "w") as metadata_file:
        json.dump({"bounds":{"xmin":int(x_min), "ymin":int(y_min), "zmin":int(z_min)}}, metadata_file, indent=4)
    