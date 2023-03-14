# train/test

import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from scipy import interp
import matplotlib.pyplot as plt
from  tf.keras.callbacks import LearningRateScheduler

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
	drop = 0.5
	epochs_drop = 10.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate
    
adam = Adam(learning_rate=lr_schedule)
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
