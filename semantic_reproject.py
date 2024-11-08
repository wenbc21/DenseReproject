import open3d as o3d
import numpy as np
import os
import cv2
import lxml.etree
from collections import Counter
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

def _get_kitti_360_3d_bbox_annotations(xml_node):
    bbox3d = None
    transform = _get_kitti_360_annotation_matrix(xml_node.find("transform"))
    vertices = _get_kitti_360_annotation_matrix(xml_node.find("vertices"))
    R = transform[:3, :3]
    t = transform[:3, 3]
    bbox3d = {
        "name": xml_node.tag,
        "vertices": np.matmul(R, vertices.transpose()).transpose() + t,
    }

    return bbox3d

def _get_kitti_360_annotation_matrix(xml_node):
    # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/annotation.py#L111-L123
    rows = int(xml_node.find("rows").text)
    cols = int(xml_node.find("cols").text)
    data = xml_node.find("data").text.split(" ")
    mat = []
    for d in data:
        d = d.replace("\n", "")
        if len(d) < 1:
            continue
        mat.append(float(d))

    mat = np.reshape(mat, [rows, cols])
    return mat

def load_intrinsics(intrinsics_fn):
    with open(intrinsics_fn, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = line.split(' ')
            if line[0] == 'P_rect_00:':
                P_rect_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'P_rect_01:':
                P_rect_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'R_rect_00:':
                R_rect_00 = np.array(line[1:], dtype=np.float32).reshape(3, 3)
            elif line[0] == 'R_rect_01:':
                R_rect_01 = np.array(line[1:], dtype=np.float32).reshape(3, 3)
    return P_rect_00, P_rect_01, R_rect_00, R_rect_01

def load_cam_to_pose(cam_to_pose_fn):
    with open(cam_to_pose_fn, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            line = line.split(' ')
            if line[0] == 'image_00:':
                c2p_00 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
            elif line[0] == 'image_01:':
                c2p_01 = np.array(line[1:], dtype=np.float32).reshape(3, 4)
    return c2p_00, c2p_01

if __name__ == '__main__':

    DRIVE = '2013_05_28_drive_0003_sync'
    seq = "seq_003"
    root_dir = './KITTI_to_colmap/KITTI-colmap'
    data_dir = f'{root_dir}/{DRIVE}/{seq}'
    semantic_dir = f"./../KITTI/KITTI-360/data_2d_semantics/train/{DRIVE}/"
    bbox_dir = f"./../KITTI/KITTI-360/data_3d_bboxes/train_full/{DRIVE}.xml"
    os.makedirs(f"colmap_dense_vis/{DRIVE}/{seq}/semantic_pcd/", exist_ok=True)
    
    img_names = sorted(os.listdir(data_dir))
    poses_fn = f'{root_dir}/data_poses/{DRIVE}/poses.txt'
    intrinsic_fn = f'{root_dir}/calibration/perspective.txt'
    cam2pose_fn = f'{root_dir}/calibration/calib_cam_to_pose.txt'
    poses = np.loadtxt(poses_fn)
    img_id = poses[:, 0].astype(np.int32)
    poses = poses[:, 1:].reshape(-1, 3, 4)
    pose_dict = {}
    for i in range(len(img_id)):
        img_name = f'{img_id[i]:010d}.png'
        pose_dict[img_name] = poses[i]
    
    P_rect_00, P_rect_01, R_rect_00_, R_rect_01_ = load_intrinsics(intrinsic_fn)
    c2p_00, c2p_01 = load_cam_to_pose(cam2pose_fn)
    c2p_00 = np.concatenate([c2p_00, np.array([[0, 0, 0, 1]])], axis=0)
    c2p_01 = np.concatenate([c2p_01, np.array([[0, 0, 0, 1]])], axis=0)
    R_rect_00 = np.eye(4)
    R_rect_00[:3, :3] = R_rect_00_
    R_rect_01 = np.eye(4)
    R_rect_01[:3, :3] = R_rect_01_
    c2w_dict = {}
    for img_name in pose_dict.keys():
        pose = pose_dict[img_name]
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
        c2w_00 = np.matmul(np.matmul(pose, c2p_00), np.linalg.inv(R_rect_00))
        c2w_01 = np.matmul(np.matmul(pose, c2p_01), np.linalg.inv(R_rect_01))
        c2w_dict[f'00_{img_name}'] = c2w_00
        c2w_dict[f'01_{img_name}'] = c2w_01
    
    pcd = o3d.io.read_point_cloud(f"KITTI_to_colmap/colmap_res/{DRIVE}/{seq}/dense/fused.ply")
    point_cloud = np.asarray(pcd.points)
    point_color = np.asarray(pcd.colors)
    point_idx = np.arange(point_cloud.shape[0])
    point_label = [[] for _ in range(point_cloud.shape[0])]
    print(point_cloud.shape, point_color.shape, point_idx.shape, len(point_label))
    
    # intrinsic
    W = 1408
    H = 376
    focal = P_rect_00[0][0]
    cx = P_rect_00[0][2]
    cy = P_rect_00[1][2]
    K = np.array([[focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]])
    
    # depth clip range
    depth_min = 0.1
    depth_max = 100.0
    
    # homogeneous extension
    ones = np.ones((point_cloud.shape[0], 1))
    points_world_homogeneous = np.hstack((point_cloud, ones))
    
    for img_ins in tqdm(img_names):
        cam_id = img_ins.split("_")[0]
        img_id = img_ins.split("_")[1].split(".")[0]
        if not os.path.exists(os.path.join(semantic_dir, f"image_{cam_id}", "semantic", f"{img_id}.png")) :
            continue
        
        # extrinsic
        extrinsic = c2w_dict[img_ins]
        extrinsic = np.linalg.inv(extrinsic) #c2w->w2c
        
        # extrinsic matrix multiply
        points_camera_homogeneous = extrinsic @ points_world_homogeneous.T

        # depth clip
        depths = points_camera_homogeneous[2, :]
        mask = (depths > depth_min) & (depths < depth_max)
        points_camera_homogeneous = points_camera_homogeneous[:, mask]
        point_color_clip = point_color[mask]
        depths = depths[mask]
        point_idx_clip = point_idx[mask]

        # perspective projection
        points_camera = points_camera_homogeneous[:3, :] / points_camera_homogeneous[2, :]

        # intrinsic matrix multiply
        points_image_homogeneous = K @ points_camera

        # normalize
        points_image = points_image_homogeneous[:2, :] / points_image_homogeneous[2, :]
        
        # reserve points inside frustum
        visible_point = (points_image[0, :] >= 0) & (points_image[0, :] < W) & \
             (points_image[1, :] >= 0) & (points_image[1, :] < H)
        points_image = points_image[:, visible_point]
        point_color_clip = point_color_clip[visible_point]
        depths = depths[visible_point]

        # sort depth
        sorted_indices = np.argsort(depths)[::-1]
        points_image_sorted = points_image[:, sorted_indices]
        point_color_sorted = point_color_clip[sorted_indices]
        point_idx_sorted = point_idx_clip[sorted_indices]
        depth_sorted = depths[sorted_indices]
        
        # load semantic label
        semantic_img = os.path.join(semantic_dir, f"image_{cam_id}", "semantic", f"{img_id}.png")
        semantic_img = cv2.imread(semantic_img, cv2.IMREAD_GRAYSCALE)
        
        # pseudo rasterize
        index_image = np.zeros((H, W), dtype=int)
        for i in range(points_image_sorted.shape[1]):
            point = points_image_sorted[:, i]
            x, y = int(point[0]), int(point[1])
            index_image[y,x] = point_idx_sorted[i]
        for yy in range(0, H) :
            for xx in range(0, W) :
                semantic_label = semantic_img[yy,xx]
                if semantic_label in label_color_dict :
                    point_label[index_image[yy,xx]].append(semantic_label)
    
    # reassign label
    point_cloud_processed = []
    point_semantic_color = []
    for point_ins in range(point_cloud.shape[0]) :
        semantic_list = point_label[point_ins]
        if semantic_list != [] :
            count = Counter(semantic_list)
            most_common = count.most_common(1)
            if most_common[0][0] != 23 :
                semantic_color = label_color_dict[most_common[0][0]]
                point_cloud_processed.append(point_cloud[point_ins])
                point_semantic_color.append(semantic_color)
    
    point_cloud_processed = np.array(point_cloud_processed)
    point_semantic_color = np.array(point_semantic_color)
    print(point_cloud_processed.shape, point_semantic_color.shape)
    
    # remove dynamic cars
    car_points = np.all(point_semantic_color == (0, 0, 142), axis=1)
    car_point_indices = np.where(car_points)[0]
    point_cloud_car = point_cloud_processed[car_points]
    xml_root = lxml.etree.parse(bbox_dir).getroot()
    annotations = []
    reserve_id = np.zeros((point_cloud_car.shape[0]))
    for c in tqdm(xml_root, leave=False):
        if c.find("transform") is None:
            continue
        if c.find("label").text != "car" :
            continue
        if c.find("timestamp").text != "-1" :
            continue
        bbox_3d = _get_kitti_360_3d_bbox_annotations(c)
        if bbox_3d is None:
            continue
        annotations.append(bbox_3d)
    
    for anno in annotations :
        vertices = anno["vertices"]
        if vertices.shape != (8, 3) :
            continue
        bbox_center = np.mean(vertices, axis=0)
        
        z_mean = np.mean(vertices[:, 2])
        lower_vertices = vertices[vertices[:, 2] < z_mean]
        upper_vertices = vertices[vertices[:, 2] > z_mean]
        selected_point = lower_vertices[0]
        distances = np.linalg.norm(lower_vertices - selected_point, axis=1)
        
        index1, index2 = np.argsort(distances)[-2:]
        axis1 = selected_point - lower_vertices[index1]
        axis2 = selected_point - lower_vertices[index2]
        distances_upper = np.linalg.norm(upper_vertices - selected_point, axis=1)
        axis3 = upper_vertices[np.argmin(distances_upper)] - selected_point
        
        edge_vectors = np.array([axis1, axis2, axis3])
        edge_vectors_normalized = edge_vectors / np.linalg.norm(edge_vectors, axis=1)[:, np.newaxis]
        rotation_matrix = edge_vectors_normalized.T
        
        points_relative_to_center = point_cloud_car - bbox_center
        rotated_points = points_relative_to_center.dot(rotation_matrix.T)
        
        bbox_relative_to_center = vertices - bbox_center
        bbox_rotated = bbox_relative_to_center.dot(rotation_matrix.T)

        min_point = bbox_rotated[np.argmin(np.sum(bbox_rotated, axis=1))]
        max_point = bbox_rotated[np.argmax(np.sum(bbox_rotated, axis=1))]

        in_box = np.all((rotated_points >= min_point) & (rotated_points <= max_point), axis=1)

        for inb in range(point_cloud_car.shape[0]) :
            if in_box[inb] :
                reserve_id[inb] = 1
    
    remove_idx = car_point_indices[np.where(reserve_id == 0)]
    point_cloud_processed = np.delete(point_cloud_processed, remove_idx, axis=0)
    point_semantic_color = np.delete(point_semantic_color, remove_idx, axis=0)
    print(point_cloud_processed.shape, point_semantic_color.shape)
    
    # save to ply file
    semantic_pcd = o3d.geometry.PointCloud()
    semantic_pcd.points = o3d.utility.Vector3dVector(point_cloud_processed)
    semantic_pcd.colors = o3d.utility.Vector3dVector(point_semantic_color / 255)
    o3d.io.write_point_cloud(f"colmap_dense_vis/{DRIVE}/{seq}/semantic_pcd/{DRIVE}_{seq}.ply", semantic_pcd)
    