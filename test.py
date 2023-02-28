from math import ceil

import numpy as np
import scipy.io as sio
import os
from keras.utils import to_categorical


from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM, BatchNormalization
from keras.models import Sequential, Model, load_model
import tensorflow
import keras
# import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K
import time

num_classes = 1
batch_size = 32
steps_per_epoch = 6
data_rows, data_cols, num_chan = 8, 9, 12#16
data_size = (data_rows, data_cols, num_chan)
flag = 'a'
t = 6

class DataGenerator(keras.utils.Sequence):

    def __init__(self, datalist_txt, batch_size = batch_size,t = t, datasize = data_size, flag = flag):
        """
           self.list_IDs:存放所有需要训练的图片文件名的列表。
           self.labels:记录图片标注的分类信息的pandas.DataFrame数据类型，已经预先给定。
           self.batch_size:每次批量生成，训练的样本大小。
           self.img_size:训练的图片尺寸。
           self.img_dir:图片在电脑中存放的路径。


        """

        self.datalist = self.get_data_path(datalist_txt)
        self.datasize = datasize
        self.t = t
        self.flag = flag
        self.batch_size = batch_size
        self.on_epoch_end()

    def get_data_path(self, read_path):
        '''
        读取数据集所有图片路径
        :param read_path:   dataset.txt文件所在路径
        :return:            返回所有图像存储路径的列表
        '''
        with open(os.path.join(read_path), "r+", encoding="utf-8", errors="ignore") as f:
            data_list = f.read().split('\n')
            data_list.remove('')  # 因为写入程序最后一个循环会有换行，所以最后一个元素是空元素，故删去

            return data_list

    def __len__(self):
        """
           返回生成器的长度，也就是总共分批生成数据的次数。

        """

        return int(ceil(len(self.datalist) // self.batch_size))


    def __getitem__(self, index):
        """
           该函数返回每次我们需要的经过处理的数据。
        """

        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.datalist[k] for k in indices]
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y


    def on_epoch_end(self):
        """
           该函数将在训练时每一个epoch结束的时候自动执行，在这里是随机打乱索引次序以方便下一batch运行。

        """
        self.indices = np.arange((len(self.datalist)//self.batch_size) * self.batch_size)

        np.random.shuffle(self.indices)


    def __data_generation(self, list_IDs_temp):
        """
           给定文件名，生成数据。
        """
        X = np.empty((self.batch_size, self.t, *self.datasize))
        Y = np.empty((self.batch_size, num_classes), dtype=np.float32)


        for i, ID in enumerate(list_IDs_temp):
            data_ = sio.loadmat(ID)
            # X[i, ...] = data_['data']
            X[i,...] = data_['data'][...,[0,1,3,4,5,7,8,9,11,12,13,15]]
            # X[i, ...] = data_['data'][..., [3, 7, 11, 15]]
            #X[i, ...] = data_['data'][..., [0, 4, 8, 12]]
            # X[i, ...] = data_['data'][..., [0, 3,4, 7,8,11,12,15]]


            if self.flag == 'v':
                y_ = data_['valence_labels']
            else:
                y_ = data_['arousal_labels']
            # Y[i,] = to_categorical(y_, num_classes)
            Y[i,] = y_
        X = [X[:,i] for i in range(self.t)]

        return X, Y

if __name__ == "__main__":
    K.clear_session()
    testdata = DataGenerator("a19testfiles.txt")
    model = load_model("modelv28.h5")
    test_loss,test_acc = model.evaluate_generator(testdata,verbose = 1)
    print(f"Test accuracy:{test_acc:.3f}")