# Dense PointCloud from Colmap to get BEV Map for GaussianCity

### KITTI-360
0. Visualize reconstructed point cloud using pyvista [colmap_dense_vis](colmap_dense_vis.py)
1. Visualize reconstructed point cloud using my own renderer [my_pointcloud_render](my_pointcloud_render.py)
2. Visualize rendered RGB images with semantic labels [fuse_pcd_semantic](fuse_pcd_semantic.py)
3. Reproject sematic label to make a new point cloud [semantic_reproject](semantic_reproject.py)
4. Get BEV Map from reprojected point cloud and extrude BEV Points from BEV Map [get_bev_map](get_bev_map.py)
5. Visualize again [my_pointcloud_render](my_pointcloud_render.py)