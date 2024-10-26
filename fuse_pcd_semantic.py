import cv2
import os

semantic_path = "./data_2d_semantics/train/2013_05_28_drive_0009_sync"
dense_pcd_path = "./colmap_dense_vis/dense_pcd_vis"

for i in range(5) :
    seq_name = f"seq_{i + 1}"
    os.makedirs(os.path.join("colmap_dense_vis", "fuse_pcd_semantic", seq_name), exist_ok=True)
    dense_pcd_imgs = sorted(os.listdir(os.path.join(dense_pcd_path, seq_name)))

    for img_name in dense_pcd_imgs :
        cam_id = img_name.split("_")[0]
        img_id = img_name.split("_")[1].split(".")[0]
        
        semantic_img = os.path.join(semantic_path, f"image_{cam_id}", "semantic_rgb", f"{img_id}.png")
        dense_pcd_img = os.path.join(dense_pcd_path, seq_name, img_name)

        # 读取两张图像
        image1 = cv2.imread(semantic_img)
        image2 = cv2.imread(dense_pcd_img)

        # 确保两张图像的大小相同
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
            print("!!!", semantic_img, dense_pcd_img)

        # 混合参数，0.5 表示两张图像各占一半
        alpha = 0.5
        beta = 1.0 - alpha
        mixed_image = cv2.addWeighted(image1, alpha, image2, beta, 0)

        # 保存混合后的图像
        cv2.imwrite(os.path.join("colmap_dense_vis", "fuse_pcd_semantic", seq_name, img_name), mixed_image)

