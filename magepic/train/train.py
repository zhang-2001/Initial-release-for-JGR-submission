import h5py
import numpy as np
import keras, os
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import math
import matplotlib.pyplot as plt
from predata import load_samp_data, load_samp_data_guiyi, load_junyun_data

# ////////////////////////////////////////////////////////////////////
# //                          _ooOoo_                               //
# //                         o8888888o                              //
# //                         88" . "88                              //
# //                         (| ^_^ |)                              //
# //                         O\  =  /O                              //
# //                      ____/`---'\____                           //
# //                    .'  \\|     |//  `.                         //
# //                   /  \\|||  :  |||//  \                        //
# //                  /  _||||| -:- |||||-  \                       //
# //                  |   | \\\  -  /// |   |                       //
# //                  | \_|  ''\---/''  |   |                       //
# //                  \  .-\__  `-`  ___/-. /                       //
# //                ___`. .'  /--.--\  `. . ___                     //
# //              ."" '<  `.___\_<|>_/___.'  >'"".                  //
# //            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
# //            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
# //      ========`-.____`-.___\_____/___.-`____.-'========         //
# //                           `=---='                              //
# //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
# //           佛祖保佑                   效果无敌                     //
# ////////////////////////////////////////////////////////////////////

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
output_net = 'result/magpic_net_v3.h5'
output_log = 'result/magpic_net_v3.log'
data_path = '../data/traindata_1.h5'


class NetworkModel(object):
    def getnet(self):
        # 输入层
        inputs = Input((1024, 3))  # 修改输入形状

        # 编码器
        conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)  # 1024 -> 512

        conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling1D(pool_size=4)(conv2)  # 修改pool_size=5 -> 4，512 -> 128
        drop2 = Dropout(0.5)(pool2)

        conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop2)
        conv3 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling1D(pool_size=2)(conv3)  # 128 -> 64

        conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling1D(pool_size=2)(drop4)  # 64 -> 32

        # 瓶颈层
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Dropout(0.5)(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv1D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        # 解码器
        up6 = Conv1D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling1D(size=4)(drop5))  # 32 -> 128
        conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
        conv6 = Conv1D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv1D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=4)(conv6))  # 128 -> 512
        conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
        conv7 = Conv1D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv1D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv7))  # 512 -> 1024
        drop8 = Dropout(0.5)(up8)
        conv8 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop8)
        conv8 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        # 输出层
        conv9 = Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        outputs = Conv1D(2, 1, activation='sigmoid')(conv9)  # 输出形状 (1024, 2)

        model = Model(inputs=inputs, outputs=outputs)

        # 定义模型
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, networknam=output_net):
        # 读取数据集
        h1 = h5py.File(data_path, 'r')
        h5obj = {'h1': h1}
        random_idx = np.random.choice(20000, 10000, replace=False)
        sampids = [['h1', i] for i in random_idx]

        dataset = load_samp_data(h5obj, sampids)
        print(dataset['xdata'].shape)
        print(dataset['ydata'].shape)
        dataset['xdata'] = np.transpose(dataset['xdata'], (0, 2, 1))  # 调整维度顺序
        dataset['ydata'] = np.transpose(dataset['ydata'], (0, 2, 1))  # 调整维度顺序
        print('dataset shape', dataset['xdata'].shape, dataset['ydata'].shape)
        model = self.getnet()
        print("got net")
        model_checkpoint = ModelCheckpoint(networknam, monitor='val_loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        hist = model.fit(dataset['xdata'], dataset['ydata'], batch_size=8, epochs=100, verbose=1, validation_split=0.1,
                         shuffle=True, callbacks=[model_checkpoint])
        # model.save('test.h5')
        with open(output_log, 'w') as f:
            f.write(str(hist.history))


if __name__ == '__main__':
    # 创建模型并打印结构
    network = NetworkModel()
    network.train()
    # model = network.getnet()
    # model.summary()

