import math
import random
import h5py
from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
import scipy.io as sio
from predatatool import linear_mapping, load_samp_data, load_txt, read_txt, loadtestdata
import os

from predatatool import load_samp_data_guiyi

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 禁用 GPU，强制使用 CPU

file_path = '../data/testdata.h5'
model_path = '../train/result/magpic_net_v3.h5'

def predict():
    print('load testing samples.')
    h1 = h5py.File(file_path, 'r')
    h5obj = {'h1': h1}
    random_idx = random.sample(range(15000, 25000), 10000)
    # random_idx = range(20000, 20501)
    sampids = [['h1', i] for i in random_idx]


    tmp = load_samp_data(h5obj, sampids)
    xdata=np.transpose(tmp['xdata'], (0, 2, 1))  # 调整维度顺序
    ydata=np.transpose(tmp['ydata'], (0, 2, 1))  # 调整维度顺序
    mag_true = tmp['magn']
    dis_true = tmp['dist']
    log_factors= tmp['log_factor']
    print(mag_true.shape, dis_true.shape)
    print('xdata,ydata',xdata.shape,ydata.shape)
    print('load trained network model.')
    model=load_model(model_path)
    print('magnitude prediction.')
    img_predict = model.predict(xdata, batch_size=1, verbose=1)
    print('output magnitude results.',img_predict.shape)
    mag_predict, dis_predict = np.split(img_predict, 2, axis=2)
    print('mag_predict,dis_predict',mag_predict.shape,dis_predict.shape)
    mag_indexes = np.argmax(mag_predict, axis=1)
    dis_indexes = np.argmax(dis_predict, axis=1)
    print('max_indexes',mag_indexes.shape)
    mag_predict = []
    dis_predict = []
    for i in range(len(mag_indexes)):
        # temp1 = linear_mapping(0, 1024, 150, 1050, mag_indexes[i][0])
        # temp2 = linear_mapping(150, 1050, 0, 1024, temp1)
        # mags = linear_mapping(0, 1024, -2, 8, temp2)

        mags = linear_mapping(0, 1024, -2, 12, mag_indexes[i][0])
        mags = mags+log_factors[i]
        mag_predict.append(mags)
    mag_loss = np.abs(mag_true - mag_predict)
    print('mag:', np.mean(mag_loss))
    for i in range(len(dis_indexes)):
        # temp1 = linear_mapping(0, 1024, 150, 1050, dis_indexes[i][0])
        # temp2 = linear_mapping(150, 1050, 0, 1024, temp1)
        # dis = linear_mapping(0, 1024, -20, 120, temp2)

        dis = linear_mapping(0, 1024, -20, 120, dis_indexes[i][0])
        dis_predict.append(dis)
    dis_loss = np.abs(dis_true - dis_predict)
    print('dis:', np.mean(dis_loss))
    mag_predict = np.array(mag_predict)
    dis_predict = np.array(dis_predict)

    #将预测结果保存到matlab中
    # mat_mag_true, mat_dis_true = np.split(ydata, 2, axis=2)
    # mat_mag_true = mat_mag_true.transpose(0, 2, 1)
    # mat_dis_true = mat_dis_true.transpose(0, 2, 1)
    # mat_mag_predict, mat_dis_predict = np.split(img_predict, 2, axis=2)
    # mat_mag_predict = mat_mag_predict.transpose(0, 2, 1)
    # mat_dis_predict = mat_dis_predict.transpose(0, 2, 1)
    # print('save predicted magnitude images.')
    # sio.savemat('matlab/predict_v1.mat', {'xdata':xdata,'mag_true':mat_mag_true,'dis_true':mat_dis_true, 'mag_predict':mat_mag_predict, 'dis_predict':mat_dis_predict})
    # #########################################################################

    load_txt(mag_true, mag_predict, mag_loss, 'mag.txt')
    load_txt(dis_true, dis_predict, dis_loss, 'dis.txt')


    # plt.scatter(mag_predict, mag_true, s=10)
    # plt.axline((0, 0), slope=1)
    # plt.xlim(0, 5)
    # plt.ylim(0, 5)
    # plt.xlabel('predict_mag')
    # plt.ylabel('true_mag')
    # # 保存图片
    # image_path = 'dis.png'  # 可以修改为你想要保存的路径和文件名
    # # plt.savefig(image_path)
    # plt.show()
    #
    #
    # plt.scatter(dis_predict, dis_true, s=10)
    # plt.axline((0, 0), slope=1)
    # plt.xlim(0, 120)
    # plt.ylim(0, 120)
    # plt.xlabel('predict_dis')
    # plt.ylabel('true_dis')
    # # 保存图片
    # image_path = 'dis.png'  # 可以修改为你想要保存的路径和文件名
    # # plt.savefig(image_path)
    # plt.show()


if __name__ == '__main__':
    predict()
