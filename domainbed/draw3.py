import numpy as np
import matplotlib.pyplot as plt


# # 示例数据
# rho_values = [1, 2, 3, 4]  # 横坐标：rho
# acc_1 = [82.346,	82.238,	84.255,	84.555]	
#   # 第一个方法的准确率
# acc_2 = [81.096,	80.849,	83.707,	84.431
# ]  # 第二个方法的准确率
# acc_3 = [81.804,	82.057,	84.292,	84.323
# ]  # 第三个方法的准确率

# # 创建一个新的图形
# plt.figure(figsize=(10, 6))

# # 绘制折线图，使用不同的颜色和标签
# plt.plot(rho_values, acc_1, label="JPS", marker='o', color='#8de0cc', linestyle='-', linewidth=2, markersize=6)
# plt.plot(rho_values, acc_2, label="Direct", marker='s', color='#89a3fe', linestyle='--', linewidth=2, markersize=6)
# plt.plot(rho_values, acc_3, label="w\o variance", marker='^', color='#bec7ff', linestyle='-.', linewidth=2, markersize=6)
# new_labels = [-1,-2,-3,-4]
# original_ticks = [1,2,3,4]

# # 设置坐标轴标签和标题
# plt.xticks(original_ticks, new_labels)
# plt.xlabel(r'$\rho$', fontsize=20)
# plt.ylabel('Accuracy', fontsize=20)
# plt.title(r'Accuracy and $\rho$', fontsize=24)

# # 设置坐标轴范围
# plt.xscale('linear')
# plt.yscale('linear')
# plt.xlim(0.0001, 0.1)
# plt.ylim(0.7, 1.0)

# 显示网格
# plt.grid(True)

# # 添加图例
# plt.legend()

# # 显示图像
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(10, 6))
# layers = [ 'layer5', 'layer6', 'layer7', 'layer8', 'layer9', 'layer10', 'layer11', 'layer12']
# values_1 = [853,678,520,542,533,283,466,853]
# values_2 = [15,16,14,18,103,114,102,157]

# # 设置条形图的宽度
# bar_width = 0.35

# # 设置位置，确保条形图不会重叠
# index = np.arange(len(layers))

# # 创建图形
# plt.bar(index, values_1, bar_width, label='Tunable Parameters', color='#d0d6ff')
# plt.bar(index + bar_width, values_2, bar_width, label='Rank', color='#fcbfba')

# # 在条形图上方显示数值
# for i in range(len(layers)):
#     plt.text(index[i], values_1[i] + 0.01, f'{values_1[i]:.2f}', ha='center', va='bottom', fontsize=16)
#     plt.text(index[i] + bar_width, values_2[i] + 0.01, f'{values_2[i]:.2f}', ha='center', va='bottom', fontsize=16)

# # 设置标签和标题
# plt.xlabel('Layers', fontsize=20)
# plt.ylabel('Numbers', fontsize=20)
# plt.title('Tunable Parameters with its Rank', fontsize=24)

# # 设置x轴的刻度标签
# plt.xticks(index + bar_width / 2, layers)

# # 添加图例
# plt.legend()

# 显示网格
# plt.grid(True)

# # 显示图形
# plt.show()

# 数据
# plt.figure(figsize=(10, 6))
# layers = ['First MLP', 'Second MLP', 'Both MLP', 'Attention', 'ALL']
# values = [84.50, 83.89, 83.47, 83.991, 82.94,]

# # 设置条形图的宽度
# bar_width = 0.6

# # 设置位置
# index = np.arange(len(layers))

# # 创建图形
# plt.bar(index, values, bar_width, label='Accuracy', color='#d0d6ff')

# # 在条形图上方显示数值
# for i in range(len(layers)):
#     plt.text(index[i], values[i] + 0.01, f'{values[i]:.2f}', ha='center', va='bottom', fontsize=16)

# # 设置标签和标题
# plt.xlabel('Layers', fontsize=20)
# plt.ylabel('Accuracy', fontsize=20)
# plt.title('Accuracy for different selected layers', fontsize=24)

# # 设置x轴的刻度标签
# plt.xticks(index, layers)
# plt.ylim(82,85)
# # 添加图例
# plt.legend()

# 显示网格
# plt.grid(True)

# # 显示图形

# 数据：每个数据集在不同状态下的参数剩余量

# datasets = ['PACS', 'VLCS', 'OfficeHome']
# original_params = [ 25200, 25200,2520]  # 原始状态的参数剩余量
# importance_params = [4263, 6097, 727]  # 重要性选择后的参数剩余量
# variance_params = [2224, 4775, 546]  # 方差选择后的参数剩余量

# # 设置条形图的宽度
# bar_width = 0.2

# # 设置x轴位置
# index = np.arange(len(datasets))

# # 创建图形
# fig, ax = plt.subplots(figsize=(10, 6))

# # 绘制三个条形图 color='#d0d6ff')
# # plt.bar(index + bar_width, values_2, bar_width, label='Rank', color='#fcbfba'
# ax.bar(index - bar_width, original_params, bar_width, label='Original', color='#dadada')
# ax.bar(index, importance_params, bar_width, label='After Importance Selection', color='#d0d6ff')
# ax.bar(index + bar_width, variance_params, bar_width, label='After Variance Selection', color='#fcbfba')

# # 在条形图上方显示数值
# for i in range(len(datasets)):
#     ax.text(index[i] - bar_width, original_params[i] + 50, f'{original_params[i]}', ha='center', va='bottom', fontsize=16)
#     ax.text(index[i], importance_params[i] + 100, f'{importance_params[i]}', ha='center', va='bottom', fontsize=16)
#     ax.text(index[i] + bar_width, variance_params[i] + 100, f'{variance_params[i]}', ha='center', va='bottom', fontsize=16)

# # 设置标签和标题
# ax.set_xlabel('Datasets', fontsize=20)
# ax.set_ylabel('Remaining Parameters', fontsize=20)
# ax.set_title('Parameter Remaining After Different Operations', fontsize=24)

# # 设置x轴刻度标签
# ax.set_xticks(index)
# ax.set_xticklabels(datasets)

# # # 添加图例
# ax.legend()
#d0d6ff')
# plt.bar(index + bar_width, values_2, bar_width, label='Rank', color='#fcbfba'
plt.figure(figsize=(10, 6))
names = ['ERM-R50', 'ERM-ViT', 'SAGM','MIRO','GESTUR',"PEGO",r'$L=12$',r'$L=8$',r'$L=4$',r'$L=2$']  # 模型名称
time_data =  [390,	500,	1000,	550,	560,500,	630,	380,	260,	210] # 时间数据，单位：秒
storage_data = [8.1,	15.3,	15.9,	17.4,	17.4,	7.1,12.2,	8.8,	4.7,	3.4]   # 存储数据，单位：MB

# 设置模型位置
y_pos = np.arange(len(names))

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制左侧的条形图（时间数据）
bar_width = 0.4  # 条形宽度
bar1 = ax1.bar(y_pos - bar_width/2, time_data, color='#d0d6ff', width=bar_width, label='Time (s)')

# 设置左侧的坐标轴
ax1.set_xlabel('Model Names',fontsize=20)
ax1.set_xticks(y_pos)
ax1.set_xticklabels(names)
ax1.set_ylabel('Time (s)',fontsize=20)
ax1.tick_params(axis='y')

# 创建右侧坐标轴
ax2 = ax1.twinx()  # 共享y轴

# 绘制右侧的条形图（存储数据）
bar2 = ax2.bar(y_pos + bar_width/2, storage_data, color='#fcbfba', width=bar_width, label='Memory (GB)')

# 设置右侧坐标轴
ax2.set_ylabel('Memory (GB)',fontsize=20)
ax2.tick_params(axis='y')

# 在条形图的顶部显示数值
for i in range(len(names)):
    ax1.text(y_pos[i] - bar_width/2, time_data[i] + 0.5, str(time_data[i]), ha='center', color='#c2c8ef',fontsize=16)
    ax2.text(y_pos[i] + bar_width/2, storage_data[i] + 0.1, str(storage_data[i]), ha='center', color='#efb5b1',fontsize=16)

# 设置标题
plt.title('Efficiency experiment',fontsize=24)

# 调整布局
plt.tight_layout()
fig.legend(handles=[bar1, bar2],bbox_to_anchor=(0.9, 0.9), borderaxespad=0.)
output_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/result/d6.png'  # 修改为你希望保存的路径
plt.savefig(output_path,dpi=200)
# # Define vectors

# v1 = np.array([2, 2, 10])  # 向量1
# v2 = np.array([2.5, 2.1, 5])  # 向量2
# v3 = np.array([1.8, 2.2, 20])  # 向量3


# # # 定义颜色
# color_domain1 = '#1f77b4'  # 蓝色
# color_domain2 = '#8de0cc'  # 橙色
# color_domain3 = '#94e494'  # 绿色
# # Calculate the sum of the vectors
# v_sum = (v1 + v2 + v3) / 3
# linestyle='-.'
# dim1,dim2 = 1,2
# # Create a new figure
# fig, ax = plt.subplots(figsize=(8, 8))

# # Plot the individual vectors as arrows
# ax.plot([0, v1[dim1]], [0, v1[dim2 ]], linestyle = linestyle, label='Doamin 1',color = color_domain1, alpha=0.7,linewidth=5)
# ax.plot([0, v2[dim1]], [0, v2[dim2 ]], linestyle = linestyle, label='Vector 2',color = color_domain2, alpha=0.7,linewidth=5)
# ax.plot([0, v3[dim1]], [0, v3[dim2 ]], linestyle = linestyle, label='Vector 3',color = color_domain3 ,alpha=0.7,linewidth=5)

# # Plot the mean vector as a line
# ax.plot([0,v_sum[dim1]], [0, v_sum[dim2 ]+0.2], color = 'gray', label='Mean Vector', alpha=0.8,linewidth=7)

# # Set the axis limits
# ax.set_xlim([0, max(v1[dim1], v2[dim1], v3[dim1], v_sum[0]) + 0.5])
# ax.set_ylim([0, max(v1[dim2 ], v2[dim2 ], v3[dim2 ], v_sum[dim2 ]) + 0.5])

# # Add axis labels
# ax.set_xlabel('Selected Parameter 2')
# ax.set_ylabel('Selected Parameter 3')

# # Add a legend


# # Make the aspect ratio equal, so the vectors are not distorted
# # ax.set_aspect('equal')

# output_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/result/vector_plot3.png'  # 修改为你希望保存的路径
# plt.savefig(output_path,dpi=200)
# # 关闭图形（避免弹出窗口）
# plt.close()