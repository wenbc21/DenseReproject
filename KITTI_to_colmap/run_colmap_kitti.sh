# Please specify SEQUENCE, PROJECT_ROOT and ROOT_DIR
# SEQUENCE: pre-processed sequence id
# PROJECT_ROOT: root directory of the colmap project to save the results
# ROOT_DIR: root directory of the dataset (pre-processed)
SEQUENCE='2013_05_28_drive_0003_sync'
PROJECT_ROOT='/media/amax/f566c04f-5a6b-4301-a323-48c9def2bf1f/amax/Project/wbc/KITTI_to_colmap/colmap_res'
ROOT_DIR='/media/amax/f566c04f-5a6b-4301-a323-48c9def2bf1f/amax/Project/wbc/KITTI_to_colmap/KITTI-colmap'
PROJECT_PATH=${PROJECT_ROOT}/${SEQUENCE}

WORK_SPACE="$PWD"

if [ ! -d ${PROJECT_PATH} ]; then
    mkdir -p ${PROJECT_PATH}
fi
cd ${PROJECT_PATH}

# colmap feature_extractor \
# --ImageReader.camera_model SIMPLE_PINHOLE  \
# --ImageReader.single_camera 1 \
# --ImageReader.camera_params 552.554261,682.049453,238.769549 \
# --database_path database.db \
# --image_path ${ROOT_DIR}/${SEQUENCE}

python ${WORK_SPACE}/colmap_kitti.py \
--project_path ${PROJECT_PATH} \
--data_path ${ROOT_DIR}

TRIANGULATED_DIR=${PROJECT_PATH}/triangulated/sparse/model
if [ ! -d ${TRIANGULATED_DIR} ]; then
    mkdir -p ${TRIANGULATED_DIR}
fi

colmap exhaustive_matcher \
--database_path database.db 
colmap point_triangulator \
--database_path database.db \
--image_path ${ROOT_DIR}/${SEQUENCE} \
--input_path created/sparse/model --output_path triangulated/sparse/model

colmap image_undistorter \
    --image_path ${ROOT_DIR}/${SEQUENCE} \
    --input_path triangulated/sparse/model \
    --output_path dense
colmap patch_match_stereo \
    --workspace_path dense
colmap stereo_fusion \
    --workspace_path dense \
    --output_path dense/fused.ply

# python ${WORK_SPACE}/preprocess/colmap_bin2npy.py \
# --project_path ${PROJECT_PATH} \
# --save_dir ${ROOT_DIR}/${SEQUENCE}

# colmap poisson_mesher \
#     --input_path dense/fused.ply \
#     --output_path dense/meshed-poisson.ply

colmap delaunay_mesher \
    --input_path dense \
    --output_path dense/meshed-delaunay.ply