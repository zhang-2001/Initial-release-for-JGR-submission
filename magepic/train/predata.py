import math

import numpy as np
import math
# def loca_img_xyz1(xyr,zr,xyz,r):#[0,0.15,64],[0,0.15,512],[2,34]   2, 34
#     img = []
#     # rtz=(100.0/12.0)**2;
#     # rtz=1.0;
#     for i in range(0, xyr[2]):
#         xy = xyr[0]+xyr[1]*i
#         tmp1 = []
#         for k in range(0, zr[2]):
#             z = zr[0]+zr[1]*k
#             ftmp=(xy-xyz[0])*(xy-xyz[0])+(z-xyz[1])*(z-xyz[1])
#             print(-0.5*ftmp/r)
#             tmp1 = tmp1+[math.exp(-0.5*ftmp/r)]
#         #tmp1.append(tmp1)
#         img.append(tmp1)
#     return img
import math


def loca_img_xyz1(xyr, zr, xyz, r):
    img = []
    max_val = 0  # 用于跟踪最大值

    for i in range(0, xyr[2]):
        xy = xyr[0] + xyr[1] * i
        tmp1 = []
        for k in range(0, zr[2]):
            z = zr[0] + zr[1] * k
            ftmp = (xy - xyz[0]) ** 2 + (z - xyz[1]) ** 2
            val = math.exp(-0.5 * ftmp / r)
            if val > max_val:
                max_val = val  # 更新最大值
            tmp1.append(val)
        img.append(tmp1)

    # 归一化处理，使最大值为1
    if max_val > 0:  # 避免除以0
        img = [[val / max_val for val in row] for row in img]

    return img


# 示例调用
distance = 2
mag = 34
y_data = loca_img_xyz1([0, 0.2, 512], [0, 0.15, 64], [distance, mag], 1.5 ** 2)






def linear_mapping(pre_a, pre_b, end_a, end_b, average):
    a = (end_b - end_a) / (pre_b - pre_a)
    b = end_a - (a * pre_a)
    return (a * average) + b

def gaussian_pdf(x, average, sigma):

    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - average) ** 2 / (2 * sigma ** 2))

def normalize_pdf(pdf, target_value):
    max_index = np.argmax(pdf)
    scale_factor = target_value / pdf[max_index]
    return pdf * scale_factor

def gen_pdf(pre_a, pre_b, end_a, end_b, average, sigma):
    mapping_average = linear_mapping(pre_a, pre_b, end_a, end_b, average)
    x = np.linspace(end_a, end_b, 1024)
    pdf = gaussian_pdf(x, mapping_average, sigma)
    normal_pdf = normalize_pdf(pdf, 1)
    return normal_pdf

def log_scale_normalization(wave_data):
    """
    wave_data: np.ndarray, shape = (1024, 3)
    """
    # 避免 log(0)，加一个极小值
    eps = 1e-6
    # 对数缩放，保留正负号
    normalized = np.sign(wave_data) * np.log1p(np.abs(wave_data) + eps)
    return normalized

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

    log_factor = np.log10(max_val_safe)

    return norm_wave, log_factor

def load_samp_data(h5obj, idxes):
    ydata = []
    xdata = []
    depth = []
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        print(dataset)
        # wave_data = np.array(dataset)
        data = []
        # print(data)
        if dataset.attrs["srcxyz"][2] <= 30:
            # for enz in wave_data:
            #     max_data = max(abs(enz))
            #     normalize_data = enz / (max_data + 0.01e-100)
            #     data.append(normalize_data)
            data = np.array(dataset)
            data , log_factor= pseudo_normalize_with_logfactor(data)
            print(log_factor)
            # print(data.shape)
            sigma = 35
            distance = dataset.attrs["distance"]
            # print(distance)
            mag = dataset.attrs["mag"]-log_factor

            # dep1 = linear_mapping(- 10.6, 40.6, 0, 1024, dep)
            # depth_pdf = gen_pdf(-2.8, 22.8, 0, 1024, dep, sigma)
            # print(distance)
            distance_pdf = gen_pdf(-20, 120, 0, 1024, distance, sigma)
            mag_pdf = gen_pdf(-2, 12, 0, 1024, mag, sigma)
            # plt.plot(depth_pdf)
            # plt.show()
            # plt.plot(distance_pdf)
            # plt.show()
            x_data = [data[0, :], data[1, :], data[2, :]]
            y_data = [mag_pdf, distance_pdf]
            # print(np.array(x_data).shape)
            xdata.append(x_data)
            ydata.append(y_data)
        # print(np.array(xdata).shape)
        # print(np.array(ydata).shape)
    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata)}

def load_junyun_data(h5obj, idxes):
    ydata = []
    xdata = []
    depth = []
    count_2_3 = 0
    count_3_4 = 0
    count_4_plus = 0
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        print(dataset)
        # wave_data = np.array(dataset)
        data = []
        # print(data)
        mag_value = dataset.attrs["mag"]
        # 检查是否已经达到所需数量
        if mag_value >= 2 and mag_value < 3 and count_2_3 < 1000:
            count_2_3 += 1
        elif mag_value >= 3 and mag_value < 4 and count_3_4 < 2000:
            count_3_4 += 1
        elif mag_value >= 4 and count_4_plus < 2000:
            count_4_plus += 1
        else:
            continue  # 跳过不需要的数据


        # for enz in wave_data:
        #     max_data = max(abs(enz))
        #     normalize_data = enz / (max_data + 0.01e-100)
        #     data.append(normalize_data)
        data = np.array(dataset)
        # print(data.shape)
        sigma = 35
        distance = dataset.attrs["distance"]
        print(distance)
        mag = dataset.attrs["mag"]

        # dep1 = linear_mapping(- 10.6, 40.6, 0, 1024, dep)
        # depth_pdf = gen_pdf(-2.8, 22.8, 0, 1024, dep, sigma)
        # print(distance)
        distance_pdf = gen_pdf(-20, 120, 0, 1024, distance, sigma)
        mag_pdf = gen_pdf(-2, 8, 0, 1024, mag, sigma)
        # plt.plot(depth_pdf)
        # plt.show()
        # plt.plot(distance_pdf)
        # plt.show()
        x_data = [data[0, :], data[1, :], data[2, :]]
        y_data = [mag_pdf, distance_pdf]
        # print(np.array(x_data).shape)
        xdata.append(x_data)
        ydata.append(y_data)
        # print(np.array(xdata).shape)
        # print(np.array(ydata).shape)
        if count_2_3 >= 1000 and count_3_4 >= 1000 and count_4_plus >= 2000:
            break
    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata)}

def load_samp_data_guiyi(h5obj, idxes):
    ydata = []
    xdata = []

    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        print(dataset)
        wave_data = np.array(dataset)
        data = []
        # print(data)
        if dataset.attrs["srcxyz"][2] <= 30:
            for enz in wave_data:
                max_data = max(abs(enz))
                normalize_data = enz / (max_data + 0.01e-100)
                data.append(normalize_data)
            data = np.array(data)
            # print(data.shape)
            sigma = 35
            distance = dataset.attrs["distance"]
            mag = dataset.attrs["mag"]

            distance_pdf = gen_pdf(-20, 120, 0, 1024, distance, sigma)
            mag_pdf = gen_pdf(-2, 8, 0, 1024, mag, sigma)
            x_data = [data[0, :], data[1, :], data[2, :]]
            y_data = [mag_pdf, distance_pdf]

            xdata.append(x_data)
            ydata.append(y_data)
    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata)}

