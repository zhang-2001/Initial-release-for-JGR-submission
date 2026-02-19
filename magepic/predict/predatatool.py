import numpy as np


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
    magn = []
    dist = []
    log_factors = []
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        print(dataset)
        data = []
        # print(data)
        if dataset.attrs["srcxyz"][2] <= 30:
            # for enz in wave_data:
            #     max_data = max(abs(enz))
            #     normalize_data = enz / (max_data + 0.01e-100)
            #     data.append(normalize_data)
            data = np.array(dataset)
            data, log_factor= pseudo_normalize_with_logfactor(data)
            sigma = 35
            dis = dataset.attrs["distance"]
            mag = dataset.attrs["mag"]
            # print(dep)
            # dep1 = linear_mapping(- 10.6, 40.6, 0, 1024, dep)
            # depth_pdf = gen_pdf(-2.8, 22.8, 0, 1024, dep, sigma)
            # print(distance)
            distance_pdf = gen_pdf(-20, 120, 0, 1024, dis, sigma)
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
            magn.append(mag)
            dist.append(dis)
            log_factors.append(log_factor)
        # print(np.array(xdata).shape)
        # print(np.array(ydata).shape)
    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata),
            'magn': np.array(magn),
            'dist': np.array(dist),
            'log_factor': np.array(log_factors)}
    # return {'xdata': np.array(xdata),
    #         'ydata': np.array(ydata),
    #         'depth': np.array(depth)}


def loadtestdata(h5obj, idxes):
    ydata = []
    xdata = []
    magn = []
    dist = []
    count_2_3 = 0
    count_3_4 = 0
    count_4_plus = 0
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        print(dataset)
        data = []
        mag_value = dataset.attrs["mag"]
        # 检查是否已经达到所需数量
        if mag_value >= 2 and mag_value < 3 and count_2_3 < 30:
            count_2_3 += 1
        elif mag_value >= 3 and mag_value < 4 and count_3_4 < 30:
            count_3_4 += 1
        elif mag_value >= 4 and count_4_plus < 40:
            count_4_plus += 1
        else:
            continue  # 跳过不需要的数据



        # for enz in wave_data:
        #     max_data = max(abs(enz))
        #     normalize_data = enz / (max_data + 0.01e-100)
        #     data.append(normalize_data)
        wave_data = np.array(dataset)
        for enz in wave_data:
            max_data = max(abs(enz))
            normalize_data = enz / (max_data + 0.01e-100)
            data.append(normalize_data)
        data = np.array(data)
        #data = np.array(dataset)
        sigma = 35
        dis = dataset.attrs["distance"]
        mag = dataset.attrs["mag"]
        # print(dep)
        # dep1 = linear_mapping(- 10.6, 40.6, 0, 1024, dep)
        # depth_pdf = gen_pdf(-2.8, 22.8, 0, 1024, dep, sigma)
        # print(distance)
        distance_pdf = gen_pdf(-20, 120, 0, 1024, dis, sigma)
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
        magn.append(mag)
        dist.append(dis)
        if count_2_3 >= 30 and count_3_4 >= 30 and count_4_plus >= 40:
            break

    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata),
            'magn': np.array(magn),
            'dist': np.array(dist)}
    # return {'xdata': np.array(xdata),
    #         'ydata': np.array(ydata),
    #         'depth': np.array(depth)}


def load_samp_data_guiyi(h5obj, idxes):
    ydata = []
    xdata = []
    magn = []
    dist = []
    for i, idx in enumerate(idxes):
        idx_nam = idx[0]
        idx_id = idx[1]
        dataset = h5obj[idx_nam].get('idx' + str(idx_id))
        # print(dataset)
        wave_data = np.array(dataset)
        data = []
        # print(data)
        if dataset.attrs["srcxyz"][2] <= 30:
            for enz in wave_data:
                max_data = max(abs(enz))
                normalize_data = enz / (max_data + 0.01e-100)
                data.append(normalize_data)
            data = np.array(data)
            sigma = 35
            dis = dataset.attrs["distance"]
            mag = dataset.attrs["mag"]
            # print(dep)
            # dep1 = linear_mapping(- 10.6, 40.6, 0, 1024, dep)
            # depth_pdf = gen_pdf(-2.8, 22.8, 0, 1024, dep, sigma)
            # print(distance)
            distance_pdf = gen_pdf(-20, 120, 0, 1024, dis, sigma)
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
            magn.append(mag)
            dist.append(dis)
        # print(np.array(xdata).shape)
        # print(np.array(ydata).shape)
    return {'xdata': np.array(xdata),
            'ydata': np.array(ydata),
            'magn': np.array(magn),
            'dist': np.array(dist)}
    # return {'xdata': np.array(xdata),
    #         'ydata': np.array(ydata),
    #         'depth': np.array(depth)}


# tmp = load_samp_data_(h5obj, sampids)
# tmp1 = check_wavedata(h5obj, sampids, 50)

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

# tmp = load_samp_data_(h5obj, sampids)
# tmp1 = check_wavedata(h5obj, sampids, 50)
