import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# 读取txt文件数据
with open('dis.txt', 'r') as f:
    lines = f.readlines()
    true_magnitude = np.array([float(x) for x in lines[0].strip().split()])
    predicted_magnitude = np.array([float(x) for x in lines[1].strip().split()])

# 计算R²
r2 = r2_score(true_magnitude, predicted_magnitude)

# 创建画布和子图，调整figsize缩小图片整体大小
fig, ax = plt.subplots(figsize=(6, 4))

for spine in ax.spines.values():
    spine.set_color('white')

# 设置背景色为灰色
ax.set_facecolor('#EAEAEA')

# 添加白色方格线
ax.grid(True, color='white', linestyle='-', linewidth=0.8, zorder=1)

# 绘制散点图
ax.scatter(true_magnitude, predicted_magnitude, color='#c7278b', alpha=0.2, zorder=2)

# 绘制对角线（y = x）
ax.plot([-10, 110], [-10, 110], 'k--', alpha=0.4, zorder=3)

# 添加R²文本框
ax.text(0.1, 0.9, f'$R^2 = {r2:.2f}$', transform=ax.transAxes, zorder=4,
        bbox=dict(facecolor='#dfbcd1', edgecolor='#dfbcd1', alpha=0.8, boxstyle='round,pad=0.5'))

# 设置坐标轴标签
ax.set_xlabel('True Distance', fontsize=11, color='#585858',fontweight='bold')
ax.set_ylabel('Predicted Distance', fontsize=11,color='#585858', fontweight='bold')

# 固定坐标轴范围
ax.set_xlim(-10, 110)
ax.set_ylim(-10, 110)

# 显示图形
plt.savefig('dis.png', dpi=600, bbox_inches='tight')
plt.show()