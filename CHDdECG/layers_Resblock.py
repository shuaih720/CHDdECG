#Resnet block
import tensorflow as tf
from tensorflow.keras import layers,Sequential,Model

class InputConv(tf.keras.layers.Layer):
    def __init__(self,filter_num,kernel_size=32,stride=2,name='conv',**kwargs):
        super(InputConv,self).__init__(name=name,**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu = tf.keras.layers.Activation('relu')

    def call(self,input):
        model1 = self.conv1(input)
        model1 = self.bn1(model1)
        model1 = self.relu(model1)
        
        return model1

class dense_lay(tf.keras.layers.Layer):
    
    def __init__(self,dim,name=''):
        super(dense_lay,self).__init__(name=name)
        self.l1 = tf.keras.layers.Dense(128)
        self.l2 = tf.keras.layers.Dense(dim)
        self.relu1 = tf.keras.layers.ReLU()

    def call(self,x):
        model = self.l1(x)
        model = self.relu1(model)
        model = self.l2(model)

        return model

class SE(layers.Layer):
    def __init__(self, filter_sq, input_channel,name=''):
        super(SE,self).__init__(name = name)
        self.filter_sq = filter_sq
        self.avepool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(filter_sq)
        self.relu = tf.keras.layers.Activation('relu')
        self.dense2 = tf.keras.layers.Dense(input_channel)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
    def call(self, inputs):
#         trans = tf.transpose(inputs, perm=[1, 0, 2])
#         print(inputs.shape)
        squeeze = self.avepool(inputs)
        excitation = self.dense1(squeeze)
        excitation = self.relu(excitation)
        excitation = self.dense2(excitation)
        excitation = self.sigmoid(excitation)
        excitation = excitation[:,tf.newaxis,:]
        scale = inputs * excitation
        
        return scale

class ResBlock(layers.Layer):
    def __init__(self,filter_num,kernel_size=32,stride=2,name=''):
        super(ResBlock,self).__init__(name=name)
        #conv+bn+relu
        self.conv1 = layers.Conv1D(filter_num,kernel_size=kernel_size,strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        #conv+bn
        self.conv2 = layers.Conv1D(filter_num,kernel_size=kernel_size,strides=stride,padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')
        
#         self.conv3 = layers.Conv1D(filter_num,kernel_size=1,strides=1,padding='same')
#         self.bn3 = layers.BatchNormalization()
#         self.relu3 = layers.Activation('relu')
#         self.seblock = SE(filter_num//2, filter_num)
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv1D(filter_num,kernel_size=1, strides=stride))
            self.downsample.add(layers.AveragePooling1D(padding='same'))
            
        else:
            self.downsample = lambda x : x
            
    def call(self,inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)   
#         out = self.relu2(out)       
#         out = self.conv3(out)
#         out = self.bn3(out)        
#         out = self.seblock(out)
        identity = self.downsample(inputs)
        output = layers.add([out,identity])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        output = self.relu2(output)
        
        return output