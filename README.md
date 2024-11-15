# Dense PointCloud from Colmap to get BEV Map

### How to use
Remember to edit the DRIVE and sequence configs in all files!
1. Preprocess KITTI raw data [preprocess_kitti](KITTI_to_colmap/preprocess_kitti.py) (1 min)
```
python KITTI_to_colmap/preprocess_kitti.py --drive_seq 3
```
2. Use COLMAP to reconstruct dense point cloud [run_colmap_kitti](KITTI_to_colmap/run_colmap_kitti.sh) (3 hours)
```
sh KITTI_to_colmap/run_colmap_kitti.sh
```
3. Reproject sematic label to make a new point cloud [semantic_reproject](semantic_reproject.py) (30 mins)
```
srun -p rtx3090_slab -n 1 --job-name=gaussiancity --gres=gpu:1 --kill-on-bad-exit=1 python semantic_reproject.py --DRIVE 2013_05_28_drive_0003_sync --seq seq_001
```
4. Get BEV Map from reprojected point cloud [get_bev_map](get_bev_map.py) (1 min)
```
python get_bev_map.py --DRIVE 2013_05_28_drive_0003_sync --seq seq_001
```
5. Extrude BEV Points from BEV Map [extrude_from_bev](extrude_from_bev.py) (3 mins)
```
python extrude_from_bev.py --DRIVE 2013_05_28_drive_0003_sync --seq seq_001
```
6. (Optional but highly recommended) Visualize reconstructed point cloud using my own renderer [pointcloud_render](pointcloud_render.py) (30 mins)
```
srun -p rtx3090_slab -n 1 --job-name=gaussiancity --gres=gpu:1 --kill-on-bad-exit=1 python pointcloud_render.py --DRIVE 2013_05_28_drive_0003_sync --seq seq_001
```
6. Merge BEV Maps from each sequence to a full BEV Map [merge_bev_map](merge_bev_map.py) (3 mins)
```
python python pointcloud_render.py --DRIVE 2013_05_28_drive_0003_sync
```