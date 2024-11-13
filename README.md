# Dense PointCloud from Colmap to get BEV Map

### How to use
Remember to edit the DRIVE and sequence configs in all files!
1. Preprocess KITTI raw data [preprocess_kitti](KITTI_to_colmap/preprocess_kitti.py) (1 min)
2. Use COLMAP to reconstruct dense point cloud [run_colmap_kitti](KITTI_to_colmap/run_colmap_kitti.sh) (3 hours)
3. Reproject sematic label to make a new point cloud [semantic_reproject](semantic_reproject.py) (30 mins)
4. Get BEV Map from reprojected point cloud [get_bev_map](get_bev_map.py) (1 min)
5. Extrude BEV Points from BEV Map [extrude_from_bev](extrude_from_bev.py) (3 mins)
6. (Optional but highly recommended) Visualize reconstructed point cloud using my own renderer [pointcloud_render](pointcloud_render.py) (30 mins)

### Visualize
You may visualize and check any intermediate results using these codes.
1. Visualize point cloud using pyvista [colmap_dense_vis](colmap_dense_vis.py)
2. Visualize point cloud using my own renderer [my_pointcloud_render](my_pointcloud_render.py)
3. Visualize rendered RGB images together with semantic labels [fuse_pcd_semantic](fuse_pcd_semantic.py)
