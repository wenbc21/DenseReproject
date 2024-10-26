import open3d as o3d
import numpy as np
import os
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
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



def get_points_from_projections(sem_map, tpd_hf, btu_hf) :
    points = []
    colors = []
    height = sem_map.shape[0]
    weight = sem_map.shape[1]
    for x in range(1, height-1) :
        for y in range(1, weight-1) :
            for k in range(btu_hf[x][y], tpd_hf[x][y] + 1) :
                
                sem_val = sem_map[x][y]
                tpd_val = tpd_hf[x][y]
                vege_col = np.array([107,142, 35])
                if k > tpd_val - 1 :
                    points.append([y, x, k])
                    colors.append(sem_val)
                    continue
                if k == btu_hf[x][y] and (sem_val == vege_col).all() :
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

    DRIVE = '2013_05_28_drive_0009_sync'
    semantic_pcd_dir = f"colmap_dense_vis/semantic_pcd"

    for idx in range(5):
        seq_name = f'seq_{idx+1}'
        print(f'Processing sequence: {seq_name}.')
        os.makedirs(f"colmap_dense_vis/bev_map", exist_ok=True)
        os.makedirs(f"colmap_dense_vis/extruded_pcd", exist_ok=True)
        
        pcd = o3d.io.read_point_cloud(os.path.join(semantic_pcd_dir, f"{seq_name}.ply"))
        point_cloud = np.asarray(pcd.points)
        point_color = np.asarray(pcd.colors)
        print(point_cloud.shape, point_color.shape)
        
        point_cloud *= 10
        x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
        y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
        z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

        tpd_hf = np.zeros((int(y_max - y_min + 2), int(x_max - x_min + 2)), dtype=np.int16)
        btu_hf = z_max * np.ones_like(tpd_hf)
        sem_map = np.zeros((int(y_max - y_min + 2), int(x_max - x_min + 2), 3), dtype=np.int16)
        
        
        depths = point_cloud[:, 2]
        sorted_indices = np.argsort(depths)
        point_cloud = point_cloud[sorted_indices]
        point_color = point_color[sorted_indices]
        
        for i in tqdm(range(point_cloud.shape[0]), leave=False):
            x, y, z = point_cloud[i]
            r, g, b = point_color[i] * 255
            _x, _y, _z = int(x - x_min), int(y - y_min), int(z - z_min)

            if tpd_hf[_y, _x] < _z:
                tpd_hf[_y, _x] = _z
            if btu_hf[_y, _x] > _z:
                btu_hf[_y, _x] = _z
            
            sem_map[_y, _x] = r, g, b
        
        # cv2.imwrite(f"colmap_dense_vis/bev_map/{seq_name}_semantic.png", sem_map)
        # cv2.imwrite(f"colmap_dense_vis/bev_map/{seq_name}_topdown.png", tpd_hf)
        # cv2.imwrite(f"colmap_dense_vis/bev_map/{seq_name}_bottomup.png", btu_hf)
        
        points, colors = get_points_from_projections(
            sem_map,
            tpd_hf.astype(int),
            btu_hf.astype(int)
        )
        
        extruded_pcd = o3d.geometry.PointCloud()
        extruded_pcd.points = o3d.utility.Vector3dVector((points + np.array([x_min, y_min, z_min])) / 10)
        extruded_pcd.colors = o3d.utility.Vector3dVector(colors / 255)
        o3d.io.write_point_cloud(f"colmap_dense_vis/extruded_pcd/{seq_name}.ply", extruded_pcd)



# def get_points_from_projections(
#     projections, classes, scales, seg_ins_relation, water_z, local_cords=None
# ):
#     # XYZ, Scale, Instance ID
#     points = np.empty((0, 5), dtype=np.int16)
#     for c, p in projections.items():
#         # Ignore bottom points for objects in the rest maps due to invisibility.
#         _points = _get_points_from_projection(
#             p, classes, scales, seg_ins_relation, local_cords, c != "REST"
#         )
#         if _points is not None:
#             points = np.concatenate((points, _points), axis=0)
#             logging.debug(
#                 "Category: %s: #Points: %d, Min/Max Value: (%d, %d)"
#                 % (c, len(_points), np.min(_points), np.max(_points))
#             )
#         # Move the water plane to -3.5m, which is aligned with CitySample.
#         if c == "REST" and "WATER" in classes:
#             points[:, 2][points[:, 4] == classes["WATER"]] = water_z

#     logging.debug("#Points: %d" % (len(points)))
#     return points


# def _get_points_from_projection(
#     projection,
#     classes,
#     scales,
#     seg_ins_relation,
#     local_cords=None,
#     include_btm_pts=True,
# ):
#     _projection = projection
#     if local_cords is not None:
#         # local_cords contains 5 points
#         # The first three points denotes the triangle of view frustum projection
#         # The last four points denotes the minimum rectangle of the view frustum projection
#         min_x = math.floor(np.min(local_cords[:, 0]))
#         max_x = math.ceil(np.max(local_cords[:, 0]))
#         min_y = math.floor(np.min(local_cords[:, 1]))
#         max_y = math.ceil(np.max(local_cords[:, 1]))
#         # Fix: negative index is not supported. Also aligned with the operations in get_local_projections()
#         if min_x < 0:
#             max_x -= min_x
#             min_x = 0
#         if min_y < 0:
#             max_y -= min_y
#             min_y = 0

#         _projection = {}
#         for c, p in projection.items():
#             # The smallest bounding box of the minimum rectangle
#             _projection[c] = np.ascontiguousarray(p[min_y:max_y, min_x:max_x]).astype(
#                 np.int16
#             )
#             if c == "PTS":
#                 mask = np.zeros_like(_projection[c], dtype=np.int16)
#                 cv2.fillPoly(
#                     mask,
#                     [np.array(local_cords - np.array([min_x, min_y]), dtype=np.int32)],
#                     1,
#                 )
#                 _projection[c] = _projection[c] * mask

#     assert np.max(_projection["INS"]) < 32768
#     points = footprint_extruder.get_points_from_projection(
#         include_btm_pts, # rest: false
#         {v: k for k, v in classes.items()},
#         scales,
#         seg_ins_relation,
#         np.ascontiguousarray(_projection["INS"].astype(np.int16)),
#         np.ascontiguousarray(_projection["TD_HF"].astype(np.int16)),
#         np.ascontiguousarray(_projection["BU_HF"].astype(np.int16)),
#         np.ascontiguousarray(_projection["PTS"].astype(bool)),
#     )
#     if points is not None and local_cords is not None:
#         # Recover the XY coordinates before cropping
#         points[:, 0] += min_x
#         points[:, 1] += min_y

#     return points.astype(np.int16) if points is not None else None
