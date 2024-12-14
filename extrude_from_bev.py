import open3d as o3d
import numpy as np
import os
import argparse
import cv2
from tqdm import tqdm
import json
from kitti_labels import label_color_dict, class_color_dict, car_palette, building_palette

def get_points_from_projections(sem_map, tpd_hf, btu_hf, is_vege) :
    points = []
    colors = []
    
    for x in tqdm(range(1, sem_map.shape[0] - 1), desc="Extruding from BEV Maps"):
        for y in range(1, sem_map.shape[1] - 1) :

            # for each point between the lowest and highest in this location
            for k in range(btu_hf[x, y], tpd_hf[x, y] + 1) :
                
                sem_val = sem_map[x, y]
                tpd_val = tpd_hf[x, y]
                
                if sem_val < 100 :
                    sem_rgb = class_color_dict[sem_val]
                elif 100 <= sem_val < 10000 :
                    sem_rgb = car_palette(sem_val) # car
                elif 10000 <= sem_val < 20000 :
                    sem_rgb = building_palette(sem_val) # building
                
                # add the top point
                if k > tpd_val - 1 :
                    points.append([y, x, k])
                    colors.append(sem_rgb)
                    continue
                # only for vegetation, add the lowest point
                if is_vege and k == btu_hf[x][y] :
                    points.append([y, x, k])
                    colors.append(sem_rgb)
                    continue
                # if not all nearby position share the same height, then add
                if np.count_nonzero(tpd_hf[x-1:x+1][y-1:y+1] == tpd_val) != 9 :
                    points.append([y, x, k])
                    colors.append(sem_rgb)
                    continue
                # if not all nearby position share the same semantic, then add
                if np.count_nonzero(sem_map[x-1:x+1][y-1:y+1] == sem_val) != 9 :
                    points.append([y, x, k])
                    colors.append(sem_rgb)
    return np.array(points), np.array(colors)


if __name__ == '__main__':

    # all the configs are here
    parser = argparse.ArgumentParser(description='Extrude from BEV Map')
    parser.add_argument('--DRIVE', type = str, default = '2013_05_28_drive_0003_sync')
    parser.add_argument('--seq', type = str, default = 'seq_001')
    parser.add_argument('--save_dir', type = str, default = './results')
    args = parser.parse_args()
    
    DRIVE = args.DRIVE
    seq = args.seq
    save_dir = f"{args.save_dir}/{DRIVE}/{seq}"
    os.makedirs(f"{save_dir}/extruded_pcd", exist_ok=True)
    
    # read BEV Map
    sem_map_vege = cv2.imread(f"{save_dir}/bev_map/VEGT-INS.png", cv2.IMREAD_UNCHANGED)
    tpd_hf_vege = cv2.imread(f"{save_dir}/bev_map/VEGT-TD_HF.png", cv2.IMREAD_UNCHANGED)
    btu_hf_vege = cv2.imread(f"{save_dir}/bev_map/VEGT-BU_HF.png", cv2.IMREAD_UNCHANGED)
    sem_map_rest = cv2.imread(f"{save_dir}/bev_map/REST-INS.png", cv2.IMREAD_UNCHANGED)
    tpd_hf_rest = cv2.imread(f"{save_dir}/bev_map/REST-TD_HF.png", cv2.IMREAD_UNCHANGED)
    btu_hf_rest = cv2.imread(f"{save_dir}/bev_map/REST-BU_HF.png", cv2.IMREAD_UNCHANGED)
    
    # get world relation position
    with open(f"{save_dir}/bev_map/metadata.json", "r") as metadata_file:
        metadata = json.load(metadata_file)
    x_min = metadata["bounds"]["xmin"]
    y_min = metadata["bounds"]["ymin"]
    z_min = metadata["bounds"]["zmin"]
    
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
    o3d.io.write_point_cloud(f"{save_dir}/extruded_pcd/{DRIVE}_{seq}.ply", extruded_pcd)
