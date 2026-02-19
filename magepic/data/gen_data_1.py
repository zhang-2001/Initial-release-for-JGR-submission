# Created on  12:44 2025/03/06

import math
import h5py
import pandas as pd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from distaz import DistAz
from prep_datatool import read_csvlines, sta_lta_ratio, detect_p_and_s_waves

csv_path = 'F:\zpc\italydata/metadata_Instance_events_v3.csv'
hdf5_path = 'F:\zpc\italydata/Instance_events_gm.hdf5'
out_file = 'traindata_1.h5'

prenam = '/data/'
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

print("打开地震目录csv文件")
with open(csv_path, 'r', encoding='utf-8') as file:
    csv_lines = read_csvlines(csv_path, 300000)
    for line in csv_lines[1:]:
        name = prenam + str(line[107])
        distaz = DistAz(stalon=float(line[6]), stalat=float(line[5]), evtlon=float(line[12]), evtlat=float(line[11]))
        dist = distaz.getDistanceKm()
        P_arrival_npts = line[48]
        S_arrival_npts = line[49]
        src_xyz = [float(line[12]), float(line[11]), float(line[13])]
        stn_xy = [float(line[6]), float(line[5])]
        Mag = float(line[21])
        Channel = str(line[4])
        Samp_rate = 40
        dataset_name.append(name)
        p_arrival.append(P_arrival_npts)
        s_arrival.append(S_arrival_npts)
        srcxyz_all.append(src_xyz)
        stnxy_all.append(stn_xy)
        distance_all.append(dist)
        mag_all.append(Mag)
        samp_rate_all.append(Samp_rate)
        channel_all.append(Channel)
    # print(dataset_name)
    # print(p_arrival)
    # print(s_arrival)
    # print(srcxyz_all)
    # print(stnxy_all)
    # print(distance_all)
    # print(delta_all)

print("打开HDF5文件")
error = []
index = 0
idx = 0
npts = 1024
group = h5py.File(hdf5_path, 'r')['data']
hf_o = h5py.File(out_file, 'w')
for i in dataset_name:
    try:
        if channel_all[index] != 'HH' or srcxyz_all[index][2] < 0 or srcxyz_all[index][2] > 30 or distance_all[
            index] > 100 or distance_all[index] < 0 or mag_all[index] < 2:
            h1 = h5py.File('a', 'r')
        dataset = group[i]
        ENZ_data = []
        max_list = []
        num = 0
        for enzdata in dataset[()]:  # 循环读取数据集的[x,y,z]内容

            Fs = 100
            new_Fs = 40
            window_time = 20

            data_numpy = np.array(enzdata)
            if np.isnan(data_numpy).any():
                h1 = h5py.File('a', 'r')


            data_detrended = enzdata - np.mean(enzdata) # 去均值
            data_detrended = signal.detrend(data_detrended, axis=0, type='linear')   # 去趋势

            # data_len = len(data_detrended)
            # window = signal.windows.hann(data_len)    # 汉宁窗
            # data_detrended_smoothed = signal.convolve(data_detrended, window, mode='same') / sum(window)    # 平滑
            # data_detrended = data_detrended - data_detrended_smoothed    # 去除低频趋势
            # std = np.std(enzdata)    # 标准差
            # data_normalized = enzdata / std    # 标准化

            f_low, f_high = 1, 10
            order = 4
            nyq = 0.5 * Fs
            low = f_low / nyq
            high = f_high / nyq
            b, a = signal.butter(order, [low, high], btype='bandpass')  # 滤波器系数
            data_filtered = signal.filtfilt(b, a, data_detrended, axis=0)  # 使用零相位滤波对标准化后的信号进行滤波处理

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
        srcxyz.append(srcxyz_all[index])
        stnxy.append(stnxy_all[index])
        samp_rate.append(samp_rate_all[index])
        distance.append(distance_all[index])
        mag.append(mag_all[index])
        tmp = hf_o.create_dataset('idx' + str(idx), data=data[idx])
        tmp.attrs['samp_rate'] = samp_rate[idx]
        tmp.attrs['npts'] = npts
        tmp.attrs['srcxyz'] = srcxyz[idx]
        tmp.attrs['stnxy'] = stnxy[idx]
        tmp.attrs['distance'] = distance[idx]
        tmp.attrs['mag'] = mag[idx]
        print('idx', idx, 'samp_rate:', samp_rate[idx], 'npts:', npts, 'srcxyz:', srcxyz[idx],
              'stnxy:', stnxy[idx], 'distance:', distance[idx], 'mag:', mag[idx])
        # print(data)
        idx += 1

    except:
        print('error:', 'index:', index, i)
        error.append(i)
    index += 1
# print(error)
hf_o.close()
