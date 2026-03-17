import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



import numpy as np
import matplotlib.pyplot as plt

# # 定义颜色
color_domain1 = '#1f77b4'  # 蓝色
color_domain2 = '#8de0cc'  # 橙色
color_domain3 = '#94e494'  # 绿色

# 创建一个新的图形
fig = plt.figure(figsize=(12, 8))

# 创建3D轴
ax = fig.add_subplot(111, projection='3d')

# 向量定义（根据要求定义梯度向量）
v1 = np.array([2, 2, 10])  # 向量1
v2 = np.array([2.5, 2.1, 5])  # 向量2
v3 = np.array([1.8, 2.2, 20])  # 向量3

# 绘制三个向量（通过quiver函数，控制箭头大小）
ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color=color_domain1, label='Domain 1', 
          linewidth=3, arrow_length_ratio=0.05 )
ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color=color_domain2, label='Domain 2', 
          linewidth=3, arrow_length_ratio=0.05)
ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color=color_domain3, label='Domain 3', 
          linewidth=3, arrow_length_ratio=0.05)

# 绘制每对坐标轴平面上的投影
# 在 xy 平面上的投影 (将 Z 分量设为零)
ax.quiver(0, 0, 0, v1[0], v1[1], 0, color=color_domain1, linestyle='--', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)
ax.quiver(0, 0, 0, v2[0], v2[1], 0, color=color_domain2, linestyle='--', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)
ax.quiver(0, 0, 0, v3[0], v3[1], 0, color=color_domain3, linestyle='--', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)

# 在 yz 平面上的投影 (将 X 分量设为零)
ax.quiver(0, 0, 0, 0, v1[1], v1[2], color=color_domain1, linestyle='-.', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)
ax.quiver(0, 0, 0, 0, v2[1], v2[2], color=color_domain2, linestyle='-.', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)
ax.quiver(0, 0, 0, 0, v3[1], v3[2], color=color_domain3, linestyle='-.', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)

# 在 zx 平面上的投影 (将 Y 分量设为零)
ax.quiver(0, 0, 0, v1[0], 0, v1[2], color=color_domain1, linestyle=':', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)
ax.quiver(0, 0, 0, v2[0], 0, v2[2], color=color_domain2, linestyle=':', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)
ax.quiver(0, 0, 0, v3[0], 0, v3[2], color=color_domain3, linestyle=':', alpha=0.8, arrow_length_ratio=0.02,linewidth=1.5)

# 设置坐标轴标签
ax.set_xlabel('Selected Parameter 1')
ax.set_ylabel('Selected Parameter 2')
ax.set_zlabel('Selected Parameter 3')

# 设置标题
ax.set_title('Gradient After Step 1 Selection')

# 调整视角，确保显示原点
ax.view_init(elev=20, azim=45) # 45

# 设置坐标轴的范围，确保平面显示从原点开始
ax.set_xlim([0, max(v1[0], v2[0], v3[0]) + 0.5])
ax.set_ylim([0, max(v1[1], v2[1], v3[1]) + 0.5])
ax.set_zlim([0, max(v1[2], v2[2], v3[2]) + 0.5])
ax.tick_params(axis='both', which='major', labelsize=5)
# 添加图例
ax.legend()

output_path = '/home/home_node7/wzb/自己算法/miro-main2/result/draw/result/vector_plot.png'  # 修改为你希望保存的路径
plt.savefig(output_path,dpi=200)

# 关闭图形（避免弹出窗口）
plt.close()