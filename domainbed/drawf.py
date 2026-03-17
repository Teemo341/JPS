import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from matplotlib.colors import LogNorm

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import clip
import re
def load_masks_and_importance(file_path):
    """
    从保存的文件中读取每个域的所有层的掩码和重要性信息。

    参数:
    - file_path: str，保存文件的路径。

    返回:
    - domain_masks: List[Dict[str, Tensor]]，每个域的掩码字典。
    - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典。
    """
    # 加载保存的数据
    combined_data = torch.load(file_path)

    domain_masks = []
    domain_importance = []

    # 遍历每个域，提取所有层的数据
    for domain_id, domain_data in combined_data.items():
        domain_masks.append(domain_data['masks'])
        domain_importance.append(domain_data['importance'])

    return domain_masks, domain_importance
# 画多个域的掩码图
# 读取保存的 domain_masks 文件

# 加载 combined_domain_mask 字典
# combined_domain_mask = torch.load('/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_masks.pth')
# 
# print("Loaded domain masks.")


# 颜色设置：使用对比鲜明的颜色
domain_colors = {'domain_0': 'tab:blue', 'domain_1': 'tab:orange', 'domain_2': 'tab:green'}

# 假设你感兴趣的层名
layer_name = "mlp"  # 替换为你感兴趣的层名





def fourier_transform_weight(weight_matrix):
    # 对二维权重矩阵进行傅里叶变换
    weight_fft = np.fft.fft2(weight_matrix)
    weight_fft_shifted = np.fft.fftshift(weight_fft)  # 将零频移至中心
    return np.abs(weight_fft_shifted)  # 返回傅里叶变换的幅度谱

def visualize_weight_with_mask(weight_matrix, domain_mask, output_filename):
    """
    对模型的二维线性层权重矩阵进行傅里叶变换，并根据0-1掩码为其不同部分着色。
    """
    # 1. 执行傅里叶变换
    weight_fft = fourier_transform_weight(weight_matrix)

    # 2. 使用掩码选择颜色
    masked_weight = np.zeros_like(weight_fft)
    masked_weight[domain_mask == 1] = weight_fft[domain_mask == 1]
    
    # 3. 绘制图像
    plt.figure(figsize=(6,6))
    plt.imshow(weight_fft, cmap='gray', interpolation='nearest')  # 使用灰度显示傅里叶变换的幅度
    plt.imshow(masked_weight, cmap='jet', alpha=0.5)  # 使用jet颜色地图显示掩码部分

    # 4. 保存图片
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
# # 遍历每个层
# for layer_name in combined_domain_mask[next(iter(combined_domain_mask))].keys():  # 遍历每个层
#     fig, ax = plt.subplots(figsize=(16, 16))  # 为每个层创建一个新的图

#     # 遍历每个域的掩码并绘制
#     for domain_idx, domain_masks in combined_domain_mask.items():
#         if layer_name in domain_masks:  # 确保该层在当前域中有掩码
#             mask_tensor = domain_masks[layer_name]
#             mask = mask_tensor.numpy().astype(bool)  # 转为布尔类型

#             # 获取掩码中为1的所有坐标
#             mask_coordinates = np.argwhere(mask == 1)
#             if mask_coordinates.ndim == 2 and mask_coordinates.shape[1] == 2:
#                 x_coords, y_coords = mask_coordinates[:, 1], mask_coordinates[:, 0]
#                 # 一次性绘制当前域的所有点（实心圆点）
#                 ax.scatter(x_coords, y_coords, 
#                            color=domain_colors.get(domain_idx, 'gray'), 
#                            alpha=0.5, s=1,  # 增加点的大小和透明度
#                            marker='o',  # 实心圆点
#                            label=f'{domain_idx} - {layer_name}')

#     # 添加图标题、标签
#     ax.set_title(f"Mask for Layer {layer_name} Across Domains", fontsize=16)
#     ax.set_xlabel('Width', fontsize=12)
#     ax.set_ylabel('Height', fontsize=12)

#     # 去重添加图例
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

#     # 保存图像，增加dpi提升清晰度
#     plt.savefig(f"/home/home_node7/wzb/自己算法/miro-main2/result/draw/layer_{layer_name}_combined_masks.png", 
#                 bbox_inches='tight', dpi=300)
#     plt.close(fig)  # 关闭图以便下一个图生成

# print("Mask visualization completed and saved.")


# 假设 `combined_domain_mask` 是已加载的掩码字典，`domain_colors` 是颜色映射字典

# color_all_ones = color_1  # 新定义的颜色，表示三个掩码都为1的位置
# color_partial_ones = color_2  # 新定义的颜色，表示只有部分掩码为1的位置
# # 遍历每个层名
# for layer_name in combined_domain_mask[next(iter(combined_domain_mask))].keys():  # 遍历每个层
#     fig, ax = plt.subplots(figsize=(16, 16))  # 为每个层创建一个新的图

#     # 合并所有域的掩码，计算 "三个域都为1" 的掩码
#     all_masks = []
#     for domain_idx, domain_masks in combined_domain_mask.items():
#         if layer_name in domain_masks:  # 确保该层在当前域中有掩码
#             all_masks.append(domain_masks[layer_name].numpy().astype(bool))
    
#     # 如果没有有效掩码，跳过
#     if not all_masks:
#         continue

#     mask_combined = np.logical_and.reduce(all_masks)  # 计算 "三个域都为1"
#     mask_any = np.logical_or.reduce(all_masks)  # 计算 "至少一个域为1"

#     # 分别获取 "三个域都为1" 和 "其他" 的坐标
#     coords_all_ones = np.argwhere(mask_combined)
#     coords_partial_ones = np.argwhere(mask_any & ~mask_combined)  # "其他" 的掩码

#     # 绘制 "三个域都为1" 的点
#     if coords_all_ones.shape[1]<2:
#         continue
#     if coords_all_ones.size > 0:
#         ax.scatter(coords_all_ones[:, 1], coords_all_ones[:, 0],
#                    color=color_all_ones, alpha=0.7, s=2, marker='o', label="All domains")

#     # 绘制 "其他" 的点
#     if coords_partial_ones.size > 0:
#         ax.scatter(coords_partial_ones[:, 1], coords_partial_ones[:, 0],
#                    color=color_partial_ones, alpha=0.7, s=2, marker='o', label="Partial domains")

#     # 添加图标题、标签
#     ax.set_title(f"Mask for Layer {layer_name} Across Domains", fontsize=16)
#     ax.set_xlabel('Width', fontsize=12)
#     ax.set_ylabel('Height', fontsize=12)

#     # 添加图例
#     ax.legend(loc='upper right', fontsize=10)

#     # 保存图像，增加dpi提升清晰度
#     plt.savefig(f"/home/home_node7/wzb/自己算法/miro-main2/result/draw/layer_{layer_name}_combined_masks_overlap.png", 
#                 bbox_inches='tight', dpi=300)
#     plt.close(fig)  # 关闭图以便下一个图生成

# print("Mask visualization completed and saved.")


# for layer_name in combined_domain_mask[next(iter(combined_domain_mask))].keys():  # 遍历每个层
#     fig, ax = plt.subplots(figsize=(16, 16))  # 为每个层创建一个新的图

#     # 合并所有域的掩码，计算 "三个域都为1" 的掩码
#     all_masks = []
#     for domain_idx, domain_masks in combined_domain_mask.items():
#         if layer_name in domain_masks:  # 确保该层在当前域中有掩码
#             all_masks.append(domain_masks[layer_name].numpy().astype(bool))
    
#     # 如果没有有效掩码，跳过
#     if not all_masks:
#         continue

#     mask_combined = np.logical_and.reduce(all_masks)  # 计算 "三个域都为1"
#     mask_any = np.logical_or.reduce(all_masks)  # 计算 "至少一个域为1"

#     # 分别获取 "三个域都为1" 和 "其他" 的坐标
#     coords_all_ones = np.argwhere(mask_combined)
#     coords_partial_ones = np.argwhere(mask_any & ~mask_combined)  # "其他" 的掩码

#     # 绘制 "三个域都为1" 的点（凸显紫色）
#     if coords_all_ones.shape[1]<2:
#         continue
#     coords_all_ones = coords_all_ones[(coords_all_ones[:, 0] < 50) & (coords_all_ones[:, 1] < 50)]
#     coords_partial_ones = coords_partial_ones[(coords_partial_ones[:, 0] < 50) & (coords_partial_ones[:, 1] < 50)]

#     # 设置显示范围为前100x100区域
#     ax.set_xlim(0, 50)
#     ax.set_ylim(0, 50)
#     if coords_all_ones.size > 0:
#         ax.scatter(coords_all_ones[:, 1], coords_all_ones[:, 0],
#                    color='purple', alpha=0.7, s=400, marker='o', label="All domains", edgecolors='black')

#     # 绘制 "其他" 的点（黄色，稍微深一些的颜色）
#     if coords_partial_ones.size > 0:
#         ax.scatter(coords_partial_ones[:, 1], coords_partial_ones[:, 0],
#                    color='orange', alpha=0.5, s=100, marker='o', label="Partial domains")

#     # 添加图标题、标签
#     ax.set_title(f"Mask for Layer {layer_name} Across Domains", fontsize=16)
#     ax.set_xlabel('Width', fontsize=12)
#     ax.set_ylabel('Height', fontsize=12)

#     # 添加图例
#     ax.legend(loc='upper right', fontsize=10)

#     # 保存图像
#     plt.savefig(f"/home/home_node7/wzb/自己算法/miro-main2/result/draw/layer_{layer_name}_highlighted.png", bbox_inches='tight')
#     plt.close(fig)  # 关闭图以便下一个图生成

# print("Mask visualization completed and saved.")


# def load_variance_data(file_path):
#     with open(file_path, "rb") as f:
#         variance_data = pickle.load(f)
#     return variance_data

# # 绘制频率直方图并保存
# def plot_variance_histograms(variance_data, output_dir="/home/home_node7/wzb/自己算法/miro-main2/result/draw"):
#     import os
#     os.makedirs(output_dir, exist_ok=True)  # 创建保存目录

#     for layer_name, variances in variance_data.items():
#         # 过滤方差大于0的部分
#         valid_variances = variances[variances > 0]
#         if valid_variances.size == 0:
#             print(f"No valid variances for layer {layer_name}, skipping.")
#             continue

#         # 计算均值
#         mean_variance = valid_variances.mean()

#         # 绘制频率直方图
#         plt.figure(figsize=(10, 6))
#         plt.hist(valid_variances, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

#         # 标注均值
#         plt.axvline(x=mean_variance, color='red', linestyle='--', label=f'Mean: {mean_variance:.4f}')
#         offset = 0.5 * mean_variance  # 可以调整这个偏移量
#         plt.text(mean_variance + offset, plt.gca().get_ylim()[1] * 0.8, f"{mean_variance:.4f}",
#                  color="red", fontsize=10, horizontalalignment='center')

#         # 图表美化
#         plt.title(f"Frequency Histogram for {layer_name}", fontsize=14)
#         plt.xlabel("Variance", fontsize=12)
#         plt.ylabel("Frequency", fontsize=12)
#         plt.legend(fontsize=12)
#         plt.grid(axis='y', linestyle='--', alpha=0.7)

#         # 保存图表
#         output_path = os.path.join(output_dir, f"{layer_name}_histogram_var.png")
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=300)
#         plt.close()

#         print(f"Saved histogram for {layer_name} to {output_path}")

# # 使用示例
# file_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/masked_grad_vars.pkl'
# variance_data = load_variance_data(file_path)
# plot_variance_histograms(variance_data)

# def load_and_plot_layer_masks_and_importance(file_path, layer_name):
#     """
#     从保存的文件中读取所有域的掩码和重要性信息，并画出指定层的掩码和重要性。
    
#     参数:
#     - file_path: str，保存掩码和重要性数据的文件路径。
#     - layer_name: str，指定要读取的层的名字。
#     """
#     # 加载保存的数据
#     data = torch.load(file_path)

#     # 创建一个图形窗口，准备绘制每个域的掩码和重要性
#     fig, axes = plt.subplots(len(data), 2, figsize=(10, len(data) * 5))

#     if len(data) == 1:
#         axes = [axes]  # 只有一个域时，确保axes仍然是二维数组

#     # 遍历每个域
#     for domain_idx, domain_data in data.items():
#         domain_mask = domain_data['masks']
#         domain_importance = domain_data['importance']

#         # 检查该域是否包含指定层
#         if layer_name in domain_mask and layer_name in domain_importance:
#             mask_tensor = domain_mask[layer_name].cpu().detach().numpy()
#             importance_tensor = domain_importance[layer_name].cpu().detach().numpy()

#             # 绘制掩码
#             axes[domain_idx][0].imshow(mask_tensor, cmap='gray')
#             axes[domain_idx][0].set_title(f'Domain {domain_idx} - Mask for {layer_name}')
#             axes[domain_idx][0].axis('off')

#             # 绘制重要性
#             axes[domain_idx][1].imshow(importance_tensor, cmap='jet')
#             axes[domain_idx][1].set_title(f'Domain {domain_idx} - Importance for {layer_name}')
#             axes[domain_idx][1].axis('off')

#     # 显示图像
#     plt.tight_layout()
#     plt.show()

# # 调用示例：
# file_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_masks_and_importance_domain_0.pth'
# layer_name = 'layer_name_1'  # 你要查找的层名称
# load_and_plot_layer_masks_and_importance(file_path, layer_name)





def process_and_save_heatmaps(domain_importance, save_dir,  k_percentage=0.2,normalize_before=True):
    """
    处理每个域的重要性张量，为每一层生成相加和相乘的结果，进行归一化并保存为热力图文件。

    参数:
    - domain_importance: List[Dict[str, Tensor]]，每个域的梯度/重要性字典列表。
    - save_dir: str，保存结果的目录。
    - normalize_before: bool，是否在计算相加或相乘之前归一化。

    返回:
    - 无，直接保存热力图文件。
    """
    # 获取所有层的名字
    all_layers = set()
    for imp in domain_importance:
        all_layers.update(imp.keys())

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每一层
    for layer_name in all_layers:
        layer_importances = []

        # 收集当前层的所有域的重要性张量
        for imp in domain_importance:
            if layer_name in imp:
                layer_tensor = imp[layer_name]
                if layer_tensor.ndim == 2 and all(dim > 1 for dim in layer_tensor.shape):
                    # 归一化（如果启用）
                    # if normalize_before:
                    #     layer_tensor = (layer_tensor - layer_tensor.min()) / (layer_tensor.max() - layer_tensor.min() + 1e-8)
                    layer_importances.append(layer_tensor)

        if not layer_importances:
            print(f"No valid 2D tensors found for layer '{layer_name}', skipping.")
            continue

        # 计算相加和相乘结果
        sum_importance = sum(layer_importances)
        prod_importance = torch.ones_like(layer_importances[0])
        for imp in layer_importances:
            prod_importance *= imp

       
        k_percentage = np.clip(k_percentage, 0, 1)  # 确保百分比在合理范围内

        # 设置文件保存路径
        sum_save_path = f"{save_dir}/{layer_name}_sum_and_prod_scatter.png"
        prod_save_path = f"{save_dir}/{layer_name}_prod_scatter.png"

        # 获取前百分之 k 的点索引
        

        # 转换为 NumPy 数组
        sum_importance = sum_importance.numpy()

        sum_importance = sum_importance[50:100,50:100]
        prod_importance = prod_importance.numpy()
        prod_importance = prod_importance[50:100,50:100]
        total_elements = sum_importance.size
        sum_importance = np.abs(sum_importance)
        prod_importance = np.abs(prod_importance)
        top_k_count = max(1, int(total_elements * k_percentage))  # 至少取 1 个点
        # 获取前 k 个点的位置（求和）
        sum_top_k_indices = np.unravel_index(np.argsort(sum_importance, axis=None)[-top_k_count:], sum_importance.shape)

        # 获取前 k 个点的位置（乘积）
        prod_top_k_indices = np.unravel_index(np.argsort(prod_importance, axis=None)[-top_k_count:], prod_importance.shape)

        # 创建散点图（求和）
        # color_1 = "#6496E6" 
        # color_2 = "#F0965A" 
        color_1 = "#50B4AA"  # 青绿色
        color_2 = "#B478C8"  # 浅紫色
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {k_percentage*100:.1f}% Sum Scatter Plot for {layer_name}")
        plt.scatter(sum_top_k_indices[1], sum_top_k_indices[0], color=color_1 , label='Sum Top K', alpha=0.7,s=100)
        plt.scatter(prod_top_k_indices[1], prod_top_k_indices[0], color=color_2, label='Product Top K', alpha=0.7,s=100)
        plt.legend()
        plt.savefig(sum_save_path)
        # plt.close()

        # # 创建散点图（乘积）
        # plt.figure(figsize=(10, 8))
        # plt.title(f"Top {k_percentage*100:.1f}% Product Scatter Plot for {layer_name}")
        
        # plt.legend()
        # plt.savefig(prod_save_path)
        # plt.close()

         # 归一化（如果不在计算前归一化，则在计算后归一化）
        # if not normalize_before:
        #     sum_importance = (sum_importance - sum_importance.min()) / (sum_importance.max() - sum_importance.min() + 1e-8)
        #     prod_importance = (prod_importance - prod_importance.min()) / (prod_importance.max() - prod_importance.min() + 1e-8)

        # 保存相加结果热力图
        # sum_save_path = f"{save_dir}/{layer_name}_sum_heatmap.png"
        # plt.figure(figsize=(10, 8))
        # sum_importance = sum_importance.numpy()
        # sum_importance = sum_importance[0:10,0:10]
        # top_k_indices = np.unravel_index(np.argsort(sum_importance, axis=None)[-10:], sum_importance.shape)
        # # sum_importance = (sum_importance - sum_importance.min()) / (sum_importance.max() - sum_importance.min() + 1e-8)
        # plt.title(f"Sum Heatmap for {layer_name}")
        # plt.imshow(sum_importance, cmap='coolwarm', interpolation='nearest',aspect='equal')
        # plt.colorbar()
        # for i in range(sum_importance.shape[0]):
        #     for j in range(sum_importance.shape[1]):
        #         value = sum_importance[i, j].item()  # 获取数值
        #         plt.text(
        #             j, i, f"{value:.3f}",  # 格式化为两位小数
        #             ha='center', va='center', color='black', fontsize=8
        #         )
        # for idx in range(10):
        #     i, j = top_k_indices[0][idx], top_k_indices[1][idx]
        #     plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))
        # # plt.tight_layout()  
        # plt.savefig(sum_save_path)
        # plt.close()

        # # 保存相乘结果热力图

        # prod_save_path = f"{save_dir}/{layer_name}_prod_heatmap.png"
        # prod_importance = prod_importance.numpy()
        # prod_importance = prod_importance[0:10,0:10]
        # top_k_indices = np.unravel_index(np.argsort(prod_importance, axis=None)[-10:], prod_importance.shape)
        # # prod_importance = (prod_importance - prod_importance.min()) / (prod_importance.max() - prod_importance.min() + 1e-8)
        # plt.figure(figsize=(10, 8))
        # plt.title(f"Product Heatmap for {layer_name}")
        # plt.imshow(prod_importance, cmap='coolwarm', interpolation='nearest',aspect='equal')
        # plt.colorbar()
        # for i in range(prod_importance.shape[0]):
        #     for j in range(prod_importance.shape[1]):
        #         value = prod_importance[i, j].item()  # 获取数值
        #         plt.text(
        #             j, i, f"{value:.3f}",  # 格式化为两位小数
        #             ha='center', va='center', color='black', fontsize=8
        #         )
        # # plt.tight_layout()  
        # for idx in range(10):
        #     i, j = top_k_indices[0][idx], top_k_indices[1][idx]
        #     plt.gca().add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='black', facecolor='none'))
        # plt.savefig(prod_save_path)
        # plt.close()

        # print(f"Heatmaps for layer '{layer_name}' saved to '{sum_save_path}' and '{prod_save_path}'.")

        print(f"Scatter plots for layer '{layer_name}' saved to '{sum_save_path}' and '{prod_save_path}'.")

def save_all_layer_mask_scatter_plots(domain_importance, save_dir, k_percentage=0.1, title_prefix="Mask Scatter Plot", figsize=(10, 8)):
    """
    遍历所有层，绘制 mask 散点图，并保存为文件，文件名根据层名称自动生成。
    
    :param domain_importance: List[Dict]，包含每个域的重要性张量，字典键为 layer_name。
    :param save_dir: str，保存的目录路径。
    :param k_percentage: float，取 top-k 百分比点，范围 [0, 1]。
    :param title_prefix: str，图的标题前缀。
    :param figsize: Tuple[int, int]，图的大小。
    """
    # 获取所有层的名字
    all_layers = set()
    for imp in domain_importance:
        all_layers.update(imp.keys())

    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每一层
    for layer_name in all_layers:
        layer_importances = []

        # 收集当前层的所有域的重要性张量
        for imp in domain_importance:
            if layer_name in imp:
                layer_tensor = imp[layer_name]
                if layer_tensor.ndim == 2 and all(dim > 1 for dim in layer_tensor.shape):
                    layer_importances.append(layer_tensor)

        if not layer_importances:
            print(f"No valid 2D tensors found for layer '{layer_name}', skipping.")
            continue

        # 将多个域的重要性张量堆叠为三维张量
        masks_stack = torch.stack(layer_importances)  # Shape: (num_masks, height, width)
        # print(masks_stack.shape)
        masks_stack =  masks_stack[:,:50,:50]

        # 计算全为1和部分为1的位置
        all_ones_mask = torch.all(masks_stack == 1, dim=0)  # 所有域中都为1的位置
        any_one_mask = torch.any(masks_stack == 1, dim=0)   # 至少一个域中为1的位置

        # 获取坐标
        partial_coords = torch.nonzero(any_one_mask, as_tuple=True)
        all_coords = torch.nonzero(all_ones_mask, as_tuple=True)
      
        

        # 绘制散点图
        plt.figure(figsize=figsize)

        # 绘制散点
        plt.scatter(
            partial_coords[1], partial_coords[0],
            color="#F28E2B", label='Partially Significant', 
            alpha=0.5, s=50,
            edgecolors='none',linewidths=0
        )
        
        # 叠加全为1的位置
        plt.scatter(
            all_coords[1], all_coords[0],
            color="#4E79A7", alpha=1.0, s=150,label = 'Fully Significant'
        )

        # 添加标题和坐标轴
        plt.title(f"{title_prefix} - {layer_name}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()

        # 保存图像到文件
        save_path = f"{save_dir}/{layer_name}_mask_scatter_plot.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Scatter plot for layer '{layer_name}' saved to {save_path}")

def fourier_transform_weight(weight_matrix):
    # 对二维权重矩阵进行傅里叶变换
    weight_fft = np.fft.fft2(weight_matrix)
    weight_fft_shifted = np.fft.fftshift(weight_fft)  # 将零频移至中心
    return np.abs(weight_fft_shifted)  # 返回傅里叶变换的幅度谱

def visualize_weight_with_mask(weight_matrix, domain_mask, output_filename):
    """
    对模型的二维线性层权重矩阵进行傅里叶变换，并根据0-1掩码为其不同部分着色。
    """
    # 1. 执行傅里叶变换
    weight_fft = fourier_transform_weight(weight_matrix)

    # 2. 使用掩码选择颜色
    masked_weight = np.zeros_like(weight_fft)
    masked_weight[domain_mask == 1] = weight_fft[domain_mask == 1]
    
    # 3. 绘制图像
    plt.figure(figsize=(6,6))
    plt.imshow(weight_fft, cmap='gray', interpolation='nearest')  # 使用灰度显示傅里叶变换的幅度
    plt.imshow(masked_weight, cmap='jet', alpha=0.5)  # 使用jet颜色地图显示掩码部分

    # 4. 保存图片
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    print('saved')

def process_all_layers(model, combined_domain_mask, output_dir):
    """
    遍历所有层并根据提供的掩码生成傅里叶变换图像，跳过bias层。
    使用正则表达式自动匹配类似 "transformer.resblocks.X.mlp.c_fc.weight" 的层。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有层并进行正则匹配
    for name, layer in model.named_modules():
    #     print(f"Layer Name: {name}") 
        for layer_name, mask in combined_domain_mask[0].items():
            # 去掉 "network." 前缀
            cleaned_layer_name = layer_name.replace('network.', '').replace('.weight', '')
            
            matched_layers = find_layers_by_pattern(model, r'transformer\.resblocks\.\d+\.mlp\.c_fc\.weight')
            

            if name == cleaned_layer_name:
                if layer is None:
                            print(f"Layer {cleaned_layer_name} not found in model.")
                            continue
                        
                if 'bias' in cleaned_layer_name:
                            continue  # 跳过bias层
                        
                weight_matrix = layer.weight.data.numpy() if hasattr(layer, 'weight') else None
                if weight_matrix is None:
                            print(f"Layer {cleaned_layer_name} does not have weight matrix.")
                            continue

                output_filename = os.path.join(output_dir, f"{cleaned_layer_name}_weight_image.png")
                visualize_weight_with_mask(weight_matrix, mask, output_filename)

def find_layers_by_pattern(model, pattern):
    """
    使用正则表达式从模型中匹配层名。
    """
    matched_layers = []
    for name, module in model.named_modules():
        if re.match(pattern, name):
            matched_layers.append(name)
    return matched_layers

def clip_imageencoder(name):
    # 加载CLIP模型并返回图像编码器
    model, _preprocess = clip.load(name, device="cpu")
    imageencoder = model.visual
    return imageencoder

color_1 = "#50B4AA"  # 青绿色
color_2 = "#B478C8"  # 浅紫色
file_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/result'
combined_domain_mask, importance = load_masks_and_importance(file_path)
model_name = 'ViT-B/16'  # 使用ViT-B/16模型
imageencoder = clip_imageencoder(model_name)

output_dir = "/home/home_node7/wzb/自己算法/miro-main2/result/draw/result"  # 设定输出目录
# 遍历字典并处理所有层
process_all_layers(imageencoder, combined_domain_mask, output_dir)
# save_all_layer_mask_scatter_plots(combined_domain_mask,save_dir)
# 示例调用
# file_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/combined_domain_r_pacs_vit.pth'
# layer_name = 'mlp'
# save_dir = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/result'
# masks, importance = load_masks_and_importance(file_path)
# process_and_save_heatmaps(importance, save_dir,normalize_before = True,k_percentage=0.05)
print('done')