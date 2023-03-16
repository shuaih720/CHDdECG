#TemporalAttention block
import tensorflow as tf
from tensorflow.keras import layers,Sequential,Model
class TemporalAttention(layers.Layer):
    def __init__(self, feature_dim,kernel_size,stride,name='',**kwargs):
        super(TemporalAttention,self).__init__(name=name,**kwargs)
        self.conv = tf.keras.layers.Conv1D(filters=feature_dim,kernel_size=kernel_size,strides=stride, padding='same')
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        
    def call(self,x):
        att = self.conv(x)
        att = self.bn(att)
        att = self.sigmoid(att)
        y = att * x
        return y