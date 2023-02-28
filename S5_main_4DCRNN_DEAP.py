from math import ceil

import numpy as np
import scipy.io as sio
import os
from keras.utils import to_categorical


from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, LeakyReLU
from keras.layers import Flatten, Dense, Concatenate, Reshape, LSTM, BatchNormalization
from keras.models import Sequential, Model
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

acc_list = []
std_list = []
all_acc = []


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

class MyCallback(keras.callbacks.Callback):
    def __init__(self, model_, train_data):
        self.mylist = []
        self.train_data = train_data
        self.model = model_
    def on_train_batch_end(self, batch, logs=None):
        self.model.summary()
        tensorflow.print("inputshape:")
        # tensorflow.print(self.model.input.shape())
        tensorflow.print(self.model.predict((traindata.__getitem__(batch))[0]).shape)
        # tensorflow.print(self.model.predict((traindata.__getitem__(batch))[0]))
        # tensorflow.print(len(traindata.__getitem__(batch)[0]))

if __name__ == '__main__':
    K.clear_session()
    start = time.time()

    if flag =='v':
        trainpath = "./a19trainfiles.txt"
        valpath = "./a19valfiles.txt"
    else:
        trainpath = "./a19trainfiles.txt"
        valpath = "./a19valfiles.txt"
    cvscores = []

    # create model
    # for train, test in kfold.split(one_falx, one_y.argmax(1)):

    def create_base_network(input_dim):
        seq = Sequential()
        # seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
        # seq.add(Conv2D(128, 4, activation='relu', padding='same', name='conv2'))
        # seq.add(Conv2D(256, 4, activation='relu', padding='same', name='conv3'))
        # seq.add(Conv2D(64, 1, activation='relu', padding='same', name='conv4'))
        seq.add(Conv2D(64, 5, activation='relu', padding='same', name='conv1', input_shape=input_dim))
        seq.add(BatchNormalization())

        # seq.add(LeakyReLU(0.05))
        seq.add(Conv2D(128, 4,  activation='relu', padding='same', name='conv2'))
        seq.add(BatchNormalization())

        # seq.add(LeakyReLU(0.05))
        seq.add(Conv2D(256, 4,  activation='relu', padding='same', name='conv3'))
        seq.add(BatchNormalization())

        # seq.add(LeakyReLU(0.05))
        seq.add(Conv2D(64, 1,  activation='relu', padding='same', name='conv4'))
        seq.add(BatchNormalization())

        # seq.add(LeakyReLU(0.05))
        seq.add(MaxPooling2D(2, 2, name='pool1'))
        seq.add(Flatten(name='fla1'))
        # seq.add(Dense(512, activation='relu', name='dense1'))
        seq.add(Dense(512, activation='relu', name='dense1'))
        seq.add(BatchNormalization())
        # seq.add(LeakyReLU(0.05))
        seq.add(Reshape((1, 512), name='reshape'))
        return seq


    base_network = create_base_network(data_size)
    input_1 = Input(shape=data_size)
    input_2 = Input(shape=data_size)
    input_3 = Input(shape=data_size)
    input_4 = Input(shape=data_size)
    input_5 = Input(shape=data_size)
    input_6 = Input(shape=data_size)

    out_all = Concatenate(axis=1)([base_network(input_1), base_network(input_2), base_network(input_3), base_network(input_4), base_network(input_5), base_network(input_6)])
    lstm_layer = LSTM(256, name='lstm')(out_all)

    out_layer = Dense(num_classes, activation='sigmoid', name='out')(lstm_layer)
    model = Model([input_1, input_2, input_3, input_4, input_5, input_6], out_layer)
    # Compile model
    model.compile(loss=keras.losses.binary_crossentropy,#keras.losses.categorical_crossentropy,binary_crossentropy
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.001,decay=0.94),#RMSprop
                  metrics=['accuracy'])

    middle = Model(inputs = base_network.get_layer('conv1').input, outputs = base_network.get_layer('conv2').output)

    # Fit the model
    traindata = DataGenerator(trainpath)
    valdata = DataGenerator(valpath)

    # print("datashape")
    # print(traindata.__getitem__(1)[0][0].shape)
    #steps_per_epoch=1,
    model.fit_generator(traindata, epochs=100, verbose=1, workers=1, validation_data = valdata) #callbacks=[MyCallback(base_network, traindata)]
    # result = middle.predict(traindata.__getitem__(1)[0][0])
    # print("input")
    # print(traindata.__getitem__(1)[0][0])
    # print("predict result:")
    # print(result)
    # print("predict resultshape:")
    # print(result.shape)
    model.save('modelv28.h5')

    end = time.time()
