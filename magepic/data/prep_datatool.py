# Created on  12:47 2025/03/04

import csv
from itertools import islice
import h5py
import numpy as np

from matplotlib import pyplot as plt
import tensorflow as tf

# h1 = h5py.File('D:\AALGX\code\Depth\data/test_noduiqi.h5', 'r')
# h5obj = {'h1': h1}
# np.random.seed(42)
# random_idx = np.random.choice(12000, 300, replace=False)
# print(random_idx)
# print(len(random_idx))
# sampids = [['h1', i] for i in random_idx]


def read_csvlines(csv_path, num):
    csv_lines = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for line in islice(reader, num):
            csv_lines.append(line)
    return csv_lines


def gaussian_pdf(x, average, sigma):

    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - average) ** 2 / (2 * sigma ** 2))


def normalize_pdf(pdf, target_value):
    max_index = np.argmax(pdf)
    scale_factor = target_value / pdf[max_index]
    return pdf * scale_factor


def linear_mapping(pre_a, pre_b, end_a, end_b, average):
    a = (end_b - end_a) / (pre_b - pre_a)
    b = end_a - (a * pre_a)
    return (a * average) + b


def gen_pdf(pre_a, pre_b, end_a, end_b, average, sigma):
    mapping_average = linear_mapping(pre_a, pre_b, end_a, end_b, average)
    x = np.linspace(pre_a, pre_b, 1024)
    pdf = gaussian_pdf(x, mapping_average, sigma)
    distance_pdf = normalize_pdf(pdf, 1)
    return distance_pdf


def load_model_and_predict(model_path, input_data):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(input_data)
    return predictions


def sta_lta_ratio(wave, fs, sta_win=0.5, lta_win=4.0):
    """
    STA/LTA 比值核心计算

    参数：
        wave : 预处理后的波形数据 (numpy数组)
        fs     : 采样率 (Hz)
        sta_win: 短时窗口长度 (秒)
        lta_win: 长时窗口长度 (秒)

    返回：
        ratio : STA/LTA 比值数组
    """
    n = len(wave)

    # 转换为样本数
    sta_samples = int(sta_win * fs)
    lta_samples = int(lta_win * fs)

    # 预计算累积和加速运算
    cumsum = np.cumsum(np.abs(wave)).astype(float)

    # 初始化结果数组
    sta = np.zeros(n)
    lta = np.zeros(n)

    # 滑动窗口计算
    for i in range(n):
        # STA 窗口范围
        sta_start = max(0, i - sta_samples + 1)
        sta_count = i - sta_start + 1

        # LTA 窗口范围
        lta_start = max(0, i - lta_samples + 1)
        lta_count = i - lta_start + 1

        # 计算窗口均值
        sta[i] = (cumsum[i] - cumsum[sta_start]) / sta_count
        lta[i] = (cumsum[i] - cumsum[lta_start]) / lta_count

    # 处理零值避免除零错误
    lta[lta == 0] = 1e-12  # 微小值代替零

    return sta / lta


def detect_p_and_s_waves(wave, fs, short_window, long_window, p_threshold, s_threshold):
    """
    检测P波和S波
        wave: 输入的地震波形信号
        short_window: 短时窗口长度
        long_window: 长时窗口长度
        p_threshold: P波检测阈值
        s_threshold: S波检测阈值
        P波和S波的索引
    """
    ratio = sta_lta_ratio(wave, fs, short_window, long_window)

    # 检测P波和S波
    p_wave_indices = np.where(ratio > p_threshold)[0]
    s_wave_indices = np.where(ratio > s_threshold)[0]

    return p_wave_indices, s_wave_indices


def check_wavedata(h5obj, idxes, step):
    xdata = []
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        # print(dataset)
        wave_data = np.array(dataset)
        data = []
        # print(data)
        for enz in wave_data:
            max_data = max(abs(enz))
            normalize_data = enz / (max_data + 0.01e-100)
            data.append(normalize_data)
        data = np.array(data)
        x_data = [data[0, :], data[1, :], data[2, :]]
        xdata.append(x_data)
    for i in range(0, len(xdata), step):
        print(i)
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(xdata[i][0])
        plt.subplot(3, 1, 2)
        plt.plot(xdata[i][1])
        plt.subplot(3, 1, 3)
        plt.plot(xdata[i][2])
        plt.show()


def delete_wavedata_and_label(data_list, label_list, list_index):
    data_mask = [False if i in list_index else True for i in range(len(data_list))]
    data_result = [data_list[i] for i in range(len(data_list)) if data_mask[i]]
    label_mask = [False if i in list_index else True for i in range(len(label_list))]
    label_result = [label_list[i] for i in range(len(label_list)) if label_mask[i]]
    return np.array(data_result), np.array(label_result)


def load_txt(list1, list2, list3, filename):
    with open(filename, 'w') as file:
        for element in list1:
            file.write(str(element) + ' ')
        file.write('\n')
        for element in list2:
            file.write(str(element.item()) + ' ')
        file.write('\n')
        for element in list3:
            file.write(str(element) + ' ')


def read_txt(filename, row, num):
    data = []
    with open(filename, 'r') as file:
        for i in range(row):
            line = file.readline().strip().split()[:num]
            line = [float(item) for item in line]
            data.append(line)
    return data


def load_samp_data_(h5obj, idxes):
    ydata = []
    xdata = []
    depth = []
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        # print(dataset)
        wave_data = np.array(dataset)
        data = []
        # print(data)
        for enz in wave_data:
            max_data = max(abs(enz))
            normalize_data = enz / (max_data + 0.01e-100)
            data.append(normalize_data)
        data = np.array(data)
        # print(data.shape)
        sigma = 2
        sigma1 = 20
        distance = dataset.attrs["distance"]
        dep = dataset.attrs["srcxyz"][2]
        dep1 = linear_mapping(0, 30, 0, 1200, dep)
        # print(distance)
        distance_pdf = gen_pdf(0, 110, 11, 99, distance, sigma)
        # plt.plot(distance_pdf)
        # plt.show()
        depth_pdf = gen_pdf(0, 1200, 150, 1050, dep1, sigma1)
        # plt.plot(depth_pdf)
        # plt.show()
        x_data = [data[0, :], data[1, :], data[2, :], distance_pdf]
        y_data = depth_pdf
        # print(np.array(x_data).shape)
        xdata.append(x_data)
        ydata.append(y_data)
        depth.append(dep)
    # print(np.array(xdata).shape)
    # print(np.array(ydata).shape)
    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata),
            'depth': np.array(depth)}


# tmp = load_samp_data_(h5obj, sampids)
# tmp1 = check_wavedata(h5obj, sampids, 50)








