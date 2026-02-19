import math
import h5py
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from distaz import DistAz

hdf5_path = '/home/data/zpc/data/wave_dist110_mag0.hdf5'
out_file = 'traindata_1.h5'

dataset_name = []
data = []
p_arrival = []
s_arrival = []
srcxyz_all = []
stnxy_all = []
samp_rate_all = []
distance_all = []
mag_all = []
channel_all = []

srcxyz = []
stnxy = []
samp_rate = []
distance = []
mag = []
channel = []

index = 0
idx = 0
npts = 1024
group = h5py.File(hdf5_path, 'r')['earthquake/local']
hf_o = h5py.File(out_file, 'w')

# 获取所有数据集名称
dataset_names = [name for name in group if isinstance(group[name], h5py.Dataset)]
for i in dataset_names[:-1]:
    datah = group[i]
    ccc = str(datah.attrs['receiver_type'])
    mmm = float(datah.attrs['source_magnitude'])
    if ccc == 'HH' and mmm > 0.1:
        dataset = np.array(datah)
        p_arrival = datah.attrs['p_arrival_sample']
        s_arrival = datah.attrs['s_arrival_sample']
        snr_db = datah.attrs['snr_db']
        source_distance_km = datah.attrs['source_distance_km']
        source_magnitude = datah.attrs['source_magnitude']
        ENZ_data = np.empty((1024, 3))
        num = 0
        for j in range(3):
            enzdata = dataset[:, j]

            Fs = 100
            new_Fs = 40
            window_time = 20

            data_numpy = np.array(enzdata)
            if np.isnan(data_numpy).any():
                h1 = h5py.File('a', 'r')

            # data_detrended = signal.detrend(enzdata, axis=0, type='linear')   # 去趋势
            # data_detrended = data_detrended - np.mean(data_detrended) # 去均值
            # data_len = len(data_detrended)
            # window = signal.windows.hann(data_len)    # 汉宁窗
            # data_detrended_smoothed = signal.convolve(data_detrended, window, mode='same') / sum(window)    # 平滑
            # data_detrended = data_detrended - data_detrended_smoothed    # 去除低频趋势
            std = np.std(enzdata)    # 标准差
            data_normalized = enzdata / std    # 标准化

            f_low, f_high = 2, 8
            order = 4
            nyq = 0.5 * Fs
            low = f_low / nyq
            high = f_high / nyq
            b, a = signal.butter(order, [low, high], btype='bandpass')  # 滤波器系数
            data_filtered = signal.filtfilt(b, a, data_normalized, axis=0)  # 使用零相位滤波对标准化后的信号进行滤波处理

            num_samples_new = int(len(data_filtered) * new_Fs / Fs)  # 计算降采样后的信号长度
            data_filtered = signal.resample(data_filtered, num_samples_new)

            # taper_window = signal.windows.hamming(len(data_filtered))   
            # data_filtered = data_filtered * taper_window #对信号应用哈明窗

            p_arr = int(p_arrival[index]) // (Fs / new_Fs)
            p_arr = int(p_arr)
            s_arr = float(s_arrival[index])
            s_arr = int(math.ceil(int(s_arr) / (Fs / new_Fs)))
            window_length = new_Fs * window_time
            if (p_arr - (new_Fs * 2)) < 0:
                h1 = h5py.File('a', 'r')
            # 定义截取窗口（P波前2秒 + 后30秒）
            window_pre = p_arr - (new_Fs * 2)
            window_end = p_arr + window_length
            data_filtered = data_filtered[window_pre: window_end]
            if len(data_filtered) < npts:
                pad_width = (0, npts - len(data_filtered))  # 表示在信号的​​左侧和右侧​​分别填充的样本数
                data_filtered = np.pad(data_filtered, pad_width=pad_width, mode="constant", constant_values=0)
            elif len(data_filtered) > npts:
                print("not enough data")
                h1 = h5py.File('a', 'r')

            # data_filtered = (data_filtered - np.mean(data_filtered)) / np.std(data_filtered)  # 标准化

            ENZ_data.append(data_filtered)
            num += 1

        data.append(ENZ_data)
        srcxyz.append(snr_db)
        samp_rate.append(40)
        distance.append(source_distance_km)
        mag.append(source_magnitude)
        tmp = hf_o.create_dataset('idx' + str(idx), data=data[idx])
        tmp.attrs['samp_rate'] = samp_rate[idx]
        tmp.attrs['npts'] = npts
        tmp.attrs['srcxyz'] = srcxyz[idx]
        tmp.attrs['distance'] = distance[idx]
        tmp.attrs['mag'] = mag[idx]
        print('idx', idx, 'samp_rate:', samp_rate[idx], 'npts:', npts, 'srcxyz:', srcxyz[idx], 'distance:', distance[idx], 'mag:', mag[idx])
        idx += 1
        index += 1
hf_o.close()





# 读取所有数据集
# datasets = {name: group[name][()] for name in dataset_names}

# 打印数据集信息
# for name, data in datasets.items():
#     print(f"数据集: {name}")
#     print(f"形状: {data.shape}")
#     print(f"数据类型: {data.dtype}")
#     print("---")
