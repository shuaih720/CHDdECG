import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from tensorflow.keras import layers,Sequential,Model
from tabnet.custom_objects import glu, GroupNormalization
import tensorflow_addons as tfa
import sklearn
import sklearn.metrics as metrics
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
import os
import tensorflow.keras as keras
from tensorflow.keras.optimizers import SGD,Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from scipy import stats
from scipy.io import loadmat
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
import random
import h5py

class InputConv(tf.keras.layers.Layer):
    def __init__(self,filter_num,kernel_size=32,stride=2,name='conv',**kwargs):
        super(InputConv,self).__init__(name=name,**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=kernel_size,strides=stride)
        self.bn1 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=filter_num,kernel_size=15,strides=2)
        self.bn2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.relu2 = tf.keras.layers.Activation('relu')

    def call(self,input):
        model1 = self.conv1(input)
        model1 = self.bn1(model1)
        model1 = self.relu1(model1)
        model1 = self.conv2(model1)
        model1 = self.bn2(model1)
        model1 = self.relu2(model1)
        
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

#Transformer block
'''Modified based on: https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/schedules/CosineDecay'''
#1.Multi-Head Attention
def scaled_dot_product_attention(q, k, v):
    '''attention(Q, K, V) = softmax(Q * K^T / sqrt(dk)) * V'''
    matmul_QK = tf.matmul(q,k,transpose_b=True)
    dk = tf.cast(tf.shape(q)[-1],tf.float32)
    scaled_attention = matmul_QK / tf.math.sqrt(dk)
    attention_weights = tf.nn.softmax(scaled_attention)  # shape=[batch_size, seq_len_q, seq_len_k]
    outputs = tf.matmul(attention_weights, v)  # shape=[batch_size, seq_len_q, depth]
    return outputs, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):       
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]    
        q = self.wq(q)  # shape=[batch_size, seq_len_q, d_model]
        k = self.wq(k)
        v = self.wq(v)
        q = self.split_heads(q, batch_size)  # shape=[batch_size, num_heads, seq_len_q, depth]
        k = self.split_heads(k, batch_size)            
        v = self.split_heads(v, batch_size)
        # scaled_attention shape=[batch_size, num_heads, seq_len_q, depth]
        # attention_weights shape=[batch_size, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # shape=[batch_size, seq_len_q, num_heads, depth]
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # shape=[batch_size, seq_len_q, d_model] 
        output = self.dense(concat_attention)
        return output, attention_weights
    
#2.Layer Normalization implementation
class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

#3. feedforward
def point_wise_feed_forward(d_model, diff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff, activation=tf.nn.relu),
        tf.keras.layers.Dense(d_model)
    ])

#4. encoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1,name=''):
        super(EncoderLayer, self).__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training):
        # multi head attention (encoder时Q = K = V)
        att_output, _ = self.mha(inputs, inputs, inputs)
        att_output = self.dropout1(att_output, training=training)
        output1 = self.layernorm1(inputs + att_output)  # shape=[batch_size, seq_len, d_model]
        # feed forward network
        ffn_output = self.ffn(output1)
        # print('3_',ffn_output.shape)
        ffn_output = self.dropout2(ffn_output, training=training)
        output2 = self.layernorm2(output1 + ffn_output)  # shape=[batch_size, seq_len, d_model]
        return output2

#Tabnet block

'''Modified based on: https://github.com/titu1994/tf-TabNet/blob/master/tabnet'''
#1.TransformBlock implementation
from tabnet.custom_objects import glu, GroupNormalization
class TransformBlock(tf.keras.layers.Layer):

    def __init__(self, features,
                 norm_type='batch',
                 momentum=0.9,
                 virtual_batch_size=None,
                 groups=2,
                 block_name='',
                 **kwargs):
        super(TransformBlock, self).__init__(**kwargs)

        self.features = features
        self.norm_type = norm_type
        self.momentum = momentum
        self.groups = groups
        self.virtual_batch_size = virtual_batch_size

        self.transform = tf.keras.layers.Dense(self.features, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                         virtual_batch_size=virtual_batch_size)

    def call(self, inputs, training=None):
        x = self.transform(inputs)
        x = self.bn(x, training=training)
        return x       

#2.Tabnet implementation
class TabNet(tf.keras.layers.Layer):
    
    def __init__(self, num_features,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=2,
                 epsilon=1e-5,
                 **kwargs):
        super(TabNet, self).__init__(**kwargs)
        
        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        if feature_dim <= output_dim:
            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        num_groups = max(1, int(num_groups))
        epsilon = float(epsilon)

        if relaxation_factor < 0.:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        if norm_type not in ['batch', 'group']:
            raise ValueError("`norm_type` must be either `batch` or `group`")

        self.feature_columns = feature_columns=None
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.norm_type = norm_type
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)

            if self.norm_type == 'batch':
                self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')
            else:
                self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')

        else:
            self.input_features = None
            self.input_bn = None

        self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f1')

        self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f2')

        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f3_{i}')
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f4_{i}')
            for i in range(self.num_decision_steps)
        ]
        
        self.transform_coef_list = [
            TransformBlock(self.num_features, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'coef_{i}')
            for i in range(self.num_decision_steps - 1)
        ]
        
        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)
            
        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.


        for ni in range(self.num_decision_steps):
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)

            if (ni > 0 or self.num_decision_steps == 1):
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

                # Decision aggregation.
                output_aggregated += decision_out

                # Aggregated masks are used for visualization of the
                # feature importance attributes.
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim:]

            if ni < (self.num_decision_steps - 1):
                mask_values = self.transform_coef_list[ni](features_for_coef, training=training)
                mask_values *= complementary_aggregated_mask_values
                mask_values = tfa.activations.sparsemax(mask_values, axis=-1)
                complementary_aggregated_mask_values *= (
                        self.relaxation_factor - mask_values)
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(
                        -mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (
                                     tf.cast(self.num_decision_steps - 1, tf.float32))
                entropy_loss = total_entropy
                masked_features = tf.multiply(mask_values, features)
                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                entropy_loss = 0.
        
        self.add_loss(self.sparsity_coefficient * entropy_loss)
        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask
        
        return output_aggregated, self._step_aggregate_feature_selection_mask

    @property
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask

#3.TabNetClassifier implementation
class TabNetClassifier(tf.keras.layers.Layer):

    def __init__(self,num_features,
                 num_classes,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,#3-5
                 relaxation_factor=1.5,#2-3
                 sparsity_coefficient=1e-5,#delete
                 norm_type='group',
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=1,
                 epsilon=1e-5,
                 **kwargs):
        super(TabNetClassifier, self).__init__(**kwargs)

        self.num_classes = num_classes

        self.tabnet = TabNet(num_features=num_features,
                             feature_dim=feature_dim,
                             output_dim=output_dim,
                             num_decision_steps=num_decision_steps,
                             relaxation_factor=relaxation_factor,
                             sparsity_coefficient=sparsity_coefficient,
                             norm_type=norm_type,
                             batch_momentum=batch_momentum,
                             virtual_batch_size=virtual_batch_size,
                             num_groups=num_groups,
                             epsilon=epsilon,
                             **kwargs)
                             

        self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')
        
    def call(self, inputs, training=None):
        out = self.tabnet(inputs, training=training)
        return out
        
    def summary(self, *super_args, **super_kwargs):
        super().summary(*super_args, **super_kwargs)
        self.tabnet.summary(*super_args, **super_kwargs)

# 4.Aliases
TabNetClassification = TabNetClassifier

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
#Sampling
class TabNet_downsampling(tf.keras.layers.Layer):
    
    def __init__(self, num_features,
                 feature_dim=64,
                 output_dim=64,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 sparsity_coefficient=1e-5,
                 norm_type='batch',#group
                 batch_momentum=0.98,
                 virtual_batch_size=None,
                 num_groups=2,
                 epsilon=1e-5,
                 **kwargs):
        
        super(TabNet_downsampling, self).__init__(**kwargs)

        if feature_dim <= output_dim:
            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        num_groups = max(1, int(num_groups))
        epsilon = float(epsilon)

        if relaxation_factor < 0.:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        if norm_type not in ['batch', 'group']:
            raise ValueError("`norm_type` must be either `batch` or `group`")

        self.feature_columns = feature_columns=None
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim     
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.norm_type = norm_type
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(f"[TabNet_lowdim]: {features_for_coeff} features will be used for decision steps.")

        if self.feature_columns is not None:
            self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)

            if self.norm_type == 'batch':
                self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum, name='input_bn')
            else:
                self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')

        else:
            self.input_features = None
            self.input_bn = None

        self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f1')

        self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type,
                                           self.batch_momentum, self.virtual_batch_size, self.num_groups,
                                           block_name='f2')

        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f3_')
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'f4_')
        ]
        
        self.transform_coef_list = [
            TransformBlock(self.num_features, self.norm_type,
                           self.batch_momentum, self.virtual_batch_size, self.num_groups, block_name=f'coef_')
        ]
        
        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)
            
        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones(
            [batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.
        if (self.num_decision_steps == 0):
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) +
                            transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[0](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) +
                            transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[0](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) +
                            transform_f3) * tf.math.sqrt(0.5)
            
            decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])
            output_aggregated += decision_out
            scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)
            aggregated_mask_values += mask_values * scale_agg
                
            features_for_coef = transform_f4[:, self.output_dim:]

        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask
        return output_aggregated,mask_values
#CHDdECG
def ECGnet(num_classes):
    #signal
    input_signal = tf.keras.Input(shape=(5000,9),name='input_signal')
        
    Signal_block_conv = InputConv(filter_num=32,kernel_size=30,stride=2,name='signal_block_conv')(input_signal)
    
#     Signal_block_res1 = ResBlock(filter_num=32,kernel_size=15,stride=2,name='signal_block_res1')(Signal_block_conv)
        
    Signal_block_res2_1 = ResBlock(filter_num=16,kernel_size=3,stride=2,name='signal_block_res2_1')(Signal_block_conv)
    Signal_block_res3_1 = ResBlock(filter_num=16,kernel_size=3,stride=2,name='signal_block_res3_1')(Signal_block_res2_1)
    Signal_block_res4_1 = ResBlock(filter_num=16,kernel_size=3,stride=2,name='signal_block_res4_1')(Signal_block_res3_1)
        
    Signal_block_res2_2 = ResBlock(filter_num=16,kernel_size=5,stride=2,name='signal_block_res2_2')(Signal_block_conv)
    Signal_block_res3_2 = ResBlock(filter_num=16,kernel_size=5,stride=2,name='signal_block_res3_2')(Signal_block_res2_2)
    Signal_block_res4_2 = ResBlock(filter_num=16,kernel_size=5,stride=2,name='signal_block_res4_2')(Signal_block_res3_2)
        
    Signal_block_res2_3 = ResBlock(filter_num=16,kernel_size=7,stride=2,name='signal_block_res2_3')(Signal_block_conv)
    Signal_block_res3_3 = ResBlock(filter_num=16,kernel_size=7,stride=2,name='signal_block_res3_3')(Signal_block_res2_3)
    Signal_block_res4_3 = ResBlock(filter_num=16,kernel_size=7,stride=2,name='signal_block_res4_3')(Signal_block_res3_3)
    
    model_con = tf.keras.layers.Concatenate(axis = -1,name = 'conv_concat')([Signal_block_res4_1,Signal_block_res4_2,Signal_block_res4_3])
    
    Signal_block_trans = EncoderLayer(d_model=48,num_heads=8, dff=128, dropout_rate=0.2,name='signal_block_trans')(model_con)
    
    Signal_temporal_attention = TemporalAttention(feature_dim=48,kernel_size=3,stride=1,name = 'signal_tem_atten')(Signal_block_trans)
    
    fla = tf.keras.layers.Flatten(name='signal_fla')(Signal_temporal_attention)
    
    Signal_block_tab1 = TabNet_downsampling(num_features=960,feature_dim=256,output_dim=128,num_decision_steps=0,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='signal_block_tab')(fla)[0]#num_features=960
    
    input_clinical = tf.keras.Input(shape=114,name='input_clinical')
    
    Clinical_block_tab1 = TabNet(num_features=114,feature_dim=256,output_dim=64,num_decision_steps=2,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='clinical_block_tab')(input_clinical)[0]
    
    input_wavelet = tf.keras.Input(shape=54,name='input_wavelet')
    
    Wavelet_block_tab1 = TabNet(num_features=54,feature_dim=256,output_dim=32,num_decision_steps=2,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='wavelet_block_tab')(input_wavelet)[0]
    
    fusion_block = tf.keras.layers.Concatenate(axis = -1,name = 'fusion_concat')([Signal_block_tab1,Clinical_block_tab1,Wavelet_block_tab1])
    
    tab1_all = TabNet(num_features=224,feature_dim=512,output_dim=256,num_decision_steps=2,relaxation_factor=3,norm_type='batch',
    batch_momentum=0.95,virtual_batch_size=None,num_groups=2,epsilon=1e-5,name='tab_con')(fusion_block)[0]
    
    fusion_block_1 = tf.keras.layers.Dense(64 , activation='relu',name='fusion_dense1')(tab1_all)
    fusion_block_2 = tf.keras.layers.Dropout(rate = 0.2,name='fusion_dropout1')(fusion_block_1)
    fusion_block_3 = tf.keras.layers.Dense(num_classes, activation='softmax',name='fusion_dense2')(fusion_block_2)
    
    model = tf.keras.models.Model([input_signal,input_clinical,input_wavelet],fusion_block_3)
    
    return model


# train/test

import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from  tensorflow.keras.callbacks import LearningRateScheduler
import math

train_input_signal = np.load('./tr_s_demo.npy')
train_input_clinical = np.load('./tr_c_demo.npy')
train_input_wavelet = np.load('./tr_w_demo.npy')
train_y = np.load('./tr_label_demo.npy')

val_input_signal = np.load('./val_s_demo.npy')
val_input_clinical = np.load('./val_c_demo.npy')
val_input_wavelet = np.load('./val_w_demo.npy')
val_y = np.load('./val_label_demo.npy')

label_smoothing=0.3
initial_learning_rate =0.01
batch_size=128
epochs=20


#model training
loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=label_smoothing,reduction=tf.keras.losses.Reduction.AUTO) 
model= CHDdECG(2)
model.summary()

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.1
	epochs_drop = 8
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
    
adam = Adam(learning_rate=0.0)
model.compile(optimizer=adam, loss=loss1, metrics=['accuracy',tf.keras.metrics.AUC(name='AUC',multi_label=True)])
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
#     model.compile(optimizer=sgd, loss=loss1, metrics=['accuracy',tf.keras.metrics.AUC(name='AUC',multi_label=True)])
#     model.compile(optimizer=adam, loss=[focal_loss(gamma=5,alpha=0.75)], metrics=['accuracy'])
history = model.fit([train_input_signal,train_input_clinical,train_input_wavelet],train_y, batch_size=batch_size, epochs=epochs,callbacks=callbacks_list,
validation_data=([val_input_signal,val_input_clinical,val_input_wavelet],val_y),verbose=1,shuffle=True,class_weight={0:0.20,1:1.}) 

# #model training
# with tf.device("/gpu:0"):    
#     loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=label_smoothing,reduction=tf.keras.losses.Reduction.AUTO) 
#     model= CHDdECG(2)
#     model.summary()
#     lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000)
#     adam = Adam(learning_rate=lr_schedule)
#     model.compile(optimizer=adam, loss=loss1, metrics=['accuracy',tf.keras.metrics.AUC(name='AUC',multi_label=True)])
# #     model.compile(optimizer=sgd, loss=loss1, metrics=['accuracy',tf.keras.metrics.AUC(name='AUC',multi_label=True)])
# #     model.compile(optimizer=adam, loss=[focal_loss(gamma=5,alpha=0.75)], metrics=['accuracy'])
#     history = model.fit([train_input_signal,train_input_clinical,train_input_wavelet],train_y, batch_size=batch_size, epochs=epochs,
#     validation_data=([val_input_signal,val_input_clinical,val_input_wavelet],val_y),verbose=1,shuffle=True,class_weight={0:0.20,1:1.})  
