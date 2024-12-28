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
    labels = []
    
    for x in tqdm(range(1, sem_map.shape[0] - 1), desc="Extruding from BEV Maps"):
        for y in range(1, sem_map.shape[1] - 1) :

            # for each point between the lowest and highest in this location
            for k in range(btu_hf[x, y], tpd_hf[x, y] + 1) :
                
                sem_val = sem_map[x, y]
                tpd_val = tpd_hf[x, y]
                
                # building roof
                if 10000 <= sem_val < 20000 and k == tpd_val:
                    sem_val += 1
                
                # add the top point
                if k > tpd_val - 1 :
                    points.append([y, x, k])
                    labels.append(sem_val)
                    continue
                # only for vegetation, add the lowest point
                if is_vege and k == btu_hf[x][y] :
                    points.append([y, x, k])
                    labels.append(sem_val)
                    continue
                # if not all nearby position share the same height, then add
                if np.count_nonzero(tpd_hf[x-1:x+1][y-1:y+1] == tpd_val) != 9 :
                    points.append([y, x, k])
                    labels.append(sem_val)
                    continue
                # if not all nearby position share the same semantic, then add
                if np.count_nonzero(sem_map[x-1:x+1][y-1:y+1] == sem_val) != 9 :
                    points.append([y, x, k])
                    labels.append(sem_val)
    return np.array(points), np.array(labels)

def compute_angle_between_planes(plane1, plane2):
    normal1 = np.array(plane1[:3])
    normal2 = np.array(plane2[:3])
    cos_theta = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


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
    points_vege, labels_vege = get_points_from_projections(
        sem_map_vege,
        tpd_hf_vege,
        btu_hf_vege, 
        True
    )
    # extrude other points from BEV Map
    points_rest, labels_rest = get_points_from_projections(
        sem_map_rest,
        tpd_hf_rest,
        btu_hf_rest, 
        False
    )
    
    # extract building points
    building_min = 10000
    building_max = np.max(labels_rest)
    
    # plane fitting for building roof
    real_building = []
    real_building_label = []
    for building_instance in range(building_min, building_max, 2) :
        
        point_cloud_facade = points_rest[labels_rest == building_instance]
        point_cloud_roof = points_rest[labels_rest == building_instance + 1]
        if point_cloud_facade.size < 50 or point_cloud_roof.size < 50 :
            continue

        roof_cloud = o3d.geometry.PointCloud()
        roof_cloud.points = o3d.utility.Vector3dVector(point_cloud_roof)

        plane_model, inliers = roof_cloud.segment_plane(distance_threshold=1.0,
                                                ransac_n=30,
                                                num_iterations=1000)
        a, b, c, d = plane_model
        
        dot_product = np.dot(np.array([a, b, c]), np.array([0, 0, 1]))
        cos_theta = dot_product / (np.linalg.norm(np.array([a, b, c])))
        angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # try to find another plane
        if angle_degrees < 10 :
            second_roof = False
        else :
            roof_left = []
            for point in point_cloud_roof:
                if a * point[0] + b * point[1] + c * point[2] + d < -0.5:
                    roof_left.append(point)
            if len(roof_left) < 50 :
                second_roof = False
            else :
                roof_cloud = o3d.geometry.PointCloud()
                roof_cloud.points = o3d.utility.Vector3dVector(np.array(roof_left))

                plane_model2, inliers = roof_cloud.segment_plane(distance_threshold=1.0,
                                                        ransac_n=30,
                                                        num_iterations=1000)
                a2, b2, c2, d2 = plane_model2
                angle = compute_angle_between_planes(plane_model, plane_model2)
                if angle < 20 :
                    second_roof = False
                else :
                    second_roof = True
        
        # reclassify building roof and facade
        building_points = np.concatenate((point_cloud_facade, point_cloud_roof), axis=0)
        
        for point in building_points:
            real_building.append(point)
            if a * point[0] + b * point[1] + c * point[2] + d >= -0.5 :
                real_building_label.append(building_instance + 1)
            elif second_roof and a2 * point[0] + b2 * point[1] + c2 * point[2] + d2 > -0.5 :
                real_building_label.append(building_instance + 1)
            else :
                real_building_label.append(building_instance)
        
    
    # merge and assign color
    real_building = np.array(real_building)
    real_building_label = np.array(real_building_label)
    points_rest = points_rest[labels_rest < 10000]
    labels_rest = labels_rest[labels_rest < 10000]
    
    points = np.concatenate((points_vege, points_rest, real_building), axis=0)
    labels = np.concatenate((labels_vege, labels_rest, real_building_label), axis=0)
    points = (points + np.array([x_min, y_min, z_min])) / 10
    
    colors = []
    for lbl in labels :
        if lbl < 100 :
            colors.append(class_color_dict[lbl])
        elif 100 <= lbl < 10000 :
            colors.append(car_palette(lbl))
        elif 10000 <= lbl < 20000 :
            colors.append(building_palette(lbl))
    colors = np.array(colors)
    
    # save extruded pcd
    extruded_pcd = o3d.geometry.PointCloud()
    extruded_pcd.points = o3d.utility.Vector3dVector(points)
    extruded_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    o3d.io.write_point_cloud(f"{save_dir}/extruded_pcd/{DRIVE}_{seq}.ply", extruded_pcd)
    