# import h5py
# import matplotlib.pyplot as plt
# import numpy as np
# from predata import load_samp_data
import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from predata import load_samp_data_guiyi, load_samp_data_buchang

# data_path = '../data/italydata.h5'

# h1 = h5py.File(data_path, 'r')
# h5obj = {'h1': h1}
# random_idx = np.random.choice(20, 1, replace=False)
# sampids = [['h1', i] for i in random_idx]
# dataset = load_samp_data(h5obj, sampids)
# dataset['xdata'] = np.transpose(dataset['xdata'], (0, 2, 1))  # 调整维度顺序
# dataset['ydata'] = np.transpose(dataset['ydata'], (0, 2, 1))  # 调整维度顺序
# print('dataset shape',dataset['xdata'].shape,dataset['ydata'].shape)
# data = dataset['ydata'].reshape(1024, 2)

# x = np.arange(1024)  # x轴为0~1023
# fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # 2行1列的子图

# # 第一条曲线
# axes[0].plot(x, data[:, 0], color="blue")
# axes[0].set_title("Mag")
# axes[0].grid(True)

# # 第二条曲线
# axes[1].plot(x, data[:, 1], color="red")
# axes[1].set_title("Distance")
# axes[1].grid(True)

# plt.tight_layout()  # 避免子图重叠
# plt.show()

# from matplotlib import pyplot as plt
# from predata import gen_pdf
#
# distance_pdf = gen_pdf(-20, 120, 0, 1024, 100, 50)
# plt.plot(distance_pdf)
# print(distance_pdf.shape)
# plt.title("Distance PDF")
# plt.xlabel("Distance")
# plt.ylabel("PDF Value")
# plt.grid(True)
# plt.show()

data_path = '../data/traindata_1.h5'
h1 = h5py.File(data_path, 'r')
h5obj = {'h1': h1}
random_idx = np.random.choice(200, 10, replace=False)
sampids = [['h1', i] for i in random_idx]
dataset = load_samp_data_buchang(h5obj, sampids)
# dataset['xdata'] = np.transpose(dataset['xdata'], (0, 2, 1))  # 调整维度顺序
data = dataset['xdata'][0]

print(data.shape)
data = np.array(data)  # 转换为NumPy数组以便于处理

# 生成模拟数据（替换为你的实际数据）
num_components = 3  # 3个分量（如E, N, Z）
num_samples = 1024
sampling_rate = 20  # 100Hz
duration = num_samples / sampling_rate  # 总时长=120秒

# 创建时间轴（单位：秒）
t = np.linspace(0, duration, num_samples, endpoint=False)

# 创建专业地震波形图
plt.figure(figsize=(10, 7))
gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

# 定义分量名称和颜色
components = ['Z', 'N', 'E']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# 计算全局y轴范围（保持相同比例）
y_min, y_max = np.min(data), np.max(data)
y_padding = (y_max - y_min) * 0.1  # 添加10%的边距

# 绘制每个分量（保持实际数据大小）
for i in range(num_components):
    ax = plt.subplot(gs[i])
    ax.plot(t, data[i], color=colors[i], linewidth=0.5, label=components[i])

    # 设置坐标轴（保持实际数据范围）
    ax.set_xlim(0, duration)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_ylabel('Amplitude (m/s²)', fontsize=10)  # 假设单位为加速度
    ax.set_title(f'{components[i]} Component', fontsize=12, pad=10)

    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    # 只在最下方子图显示x轴标签
    if i == num_components - 1:
        ax.set_xlabel('Time (seconds)', fontsize=12)
    else:
        ax.set_xticklabels([])

plt.suptitle(f'\nThree-Component Seismic Waveform ({sampling_rate}Hz sampling rate)',
             y=1.02, fontsize=14)
plt.tight_layout()
# plt.savefig('img5.png', dpi=300, bbox_inches='tight')
plt.show()