import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_heatmap(hmp, input_image=None, save_path=None, sample_idx=0, max_num=10):
    """
    可视化 heatmap 并可选叠加到原图。

    Args:
        hmp (Tensor): [B, num_classes, H, W] 预测的热力图
        input_image (Tensor): [B, 1, H, W] 原图，灰度图 (可选)
        save_path (str): 保存路径 (可选)
        sample_idx (int): 展示 batch 中第几个样本
        max_num (int): 最多显示的关键点数目
    """

    hmp = hmp.detach().cpu().numpy()  # 转为 numpy
    if input_image is not None:
        input_image = input_image.detach().cpu().numpy()

    num_classes = hmp.shape[1]
    num_classes = min(num_classes, max_num)  # 防止过多
    fig, axes = plt.subplots(1, num_classes, figsize=(3 * num_classes, 3))

    if num_classes == 1:
        axes = [axes]  # 保证可迭代

    for idx in range(num_classes):
        heatmap = hmp[sample_idx, idx, :, :]
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # 归一化

        if input_image is not None:
            # 叠加
            img = input_image[sample_idx, 0]
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            combined = 0.6 * img_norm + 0.4 * heatmap_norm  # 融合
            axes[idx].imshow(combined, cmap='jet')
        else:
            axes[idx].imshow(heatmap_norm, cmap='jet')

        axes[idx].axis('off')
        axes[idx].set_title(f'Keypoint {idx}')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

