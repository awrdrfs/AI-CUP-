import numpy as np
import tensorflow as tf
from tensorflow import keras

# 建立residual layer的class
class basicBlock(keras.layers.Layer):

    # kernel_size卷積層大小預設為3，stride為1，filter_num代表輸出多少層filter
    # 如果經過卷積層的filter_num跟原始輸入的filter_num不一樣，下面self.downsample.add會error。
    # 將change = True讓filter_num保持一致
    def __init__(self, filter_num, stride=1, change=False, kernel_size=3, padding='same'):
        super(basicBlock, self).__init__()
        
        # 內含兩個卷積層
        # padding='same'讓輸出維持19*19*X
        # 卷積層1
        self.conv1 = keras.layers.Conv2D(filter_num, kernel_size=kernel_size, strides=stride, padding=padding)
        
        #原本是先relu再BatchNormalization，改成先BatchNormalization再relu，根據cdsn的某篇文章建議(找不到了)
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        
        # 卷積層2
        self.conv2 = keras.layers.Conv2D(filter_num, kernel_size=kernel_size, strides=stride, padding=padding)
        self.bn2 = keras.layers.BatchNormalization()
        
        # 未經過卷積層的input
        if change != False:
            self.downsample = keras.Sequential()
            self.downsample.add(keras.layers.Conv2D(filter_num, kernel_size=1, strides=stride, padding=padding))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        
        #前向計算forward
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        #inputs通過identity轉換
        identity = self.downsample(inputs)
        
        # f(x)+x運算
        output = keras.layers.add([out, identity])
        
        # 再通過relu激活函數並回傳
        output = tf.nn.relu(output)
        return output

# 多個residual layer組成的block   
def build_resblock(filter_num, blocks, stride=1, change=False, kernel=3, padding='same'):
    Resblock = keras.Sequential()
    Resblock.add(basicBlock(filter_num=filter_num, stride=stride, change=change, kernel_size=kernel, padding=padding))

    #第一個之後的residual layer的stride固定是1
    for i in range(1, blocks): 
        Resblock.add(basicBlock(filter_num=filter_num, stride=1, change=change, kernel_size=kernel, padding=padding))
    return Resblock

# 建模型
def create_model():
    inputs = keras.layers.Input(shape=(19, 19, 11))
    outputs = keras.layers.Conv2D(kernel_size=3, filters=128, strides=1, padding='same')(inputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation('relu')(outputs)

    outputs = build_resblock(filter_num=128, blocks=10, change=True, kernel=3, padding='same')(outputs)
    
    outputs = keras.layers.Conv2D(kernel_size=1, filters=32, strides=1, padding='valid')(outputs)

    # 全連接層
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.BatchNormalization()(outputs)
    outputs = keras.layers.Activation('relu')(outputs)
    outputs = keras.layers.Dense(1024, activation='relu')(outputs)
    outputs = keras.layers.Dense(19*19, activation='softmax')(outputs)

    model = tf.keras.models.Model(inputs, outputs)

    return model