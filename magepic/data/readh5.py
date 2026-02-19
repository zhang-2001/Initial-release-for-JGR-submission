import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# 打开HDF5文件
file_path = 'traindata_1.h5'  # 替换为你自己的文件路径
mag = []

# with h5py.File(file_path, 'r') as file:
#     len = len(list(file.keys()))
#     for i in np.arange(0, len):
#         tmp = file['idx' + str(i)]
#         mag.append(tmp.attrs['mag'])
# sorted_mag = sorted(mag)
# #print(sorted_mag)
# with open("sorted_mag.txt", "w") as file:
#     for num in sorted_mag:
#         file.write(f"{num}\n")  # 每个数字占一行
#     print(np.amax(mag))

def pseudo_normalize_with_logfactor(wave_data):
    """
    wave_data: np.ndarray, shape = (1024, 3)

    返回:
        norm_wave: 伪归一化后的波形, shape = (1024, 3)
        log_factor: 单个归一化因子的对数 (float)
    """
    # 整个 (1024,3) 波形的最大绝对值
    max_val = np.max(np.abs(wave_data))

    # 避免除零
    max_val_safe = max_val + 1e-10

    # 伪归一化到 [-1, 1]
    norm_wave = wave_data / max_val_safe

    # 归一化因子的对数
    log_factor = np.log(max_val_safe)

    return norm_wave, log_factor
with h5py.File(file_path, 'r') as file:
    # 查看文件中包含的所有对象（数据集、组等）
    print("所有对象:")
    print(len(list(file.keys())))

    # 假设文件中有一个名为 'dataset1' 的数据集
    dataset = file['idx9999']

    # 查看数据集的内容
    print("数据集内容:")
    print(dataset[:].shape)  # 以数组的形式加载数据

    # 获取数据集的属性
    print("数据集的属性:")
    for key, value in dataset.attrs.items():
        print(f"{key}: {value}")
    data = np.array(dataset)  # 转换为NumPy数组以便于处理
    data , tt= pseudo_normalize_with_logfactor(data)

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
    # plt.savefig('img5.png', dpi=300, bbox_inches='tight')
    plt.show()
