# train/test

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,Model
from tensorflow.keras.callbacks import ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow_addons as tfa
from tabnet.custom_objects import glu, GroupNormalization
import numpy as np
import pandas as pd
import os
import random
import sklearn.metrics as metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc
from scipy import stats
from scipy.io import loadmat
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
import h5py

#model training
with tf.device("/gpu:0"):    
    loss1 = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=label_smoothing,reduction=tf.keras.losses.Reduction.AUTO) 
    model= CHDdECG(2)
    model.summary()
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_learning_rate, decay_steps=1000)
    adam = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=adam, loss=loss1, metrics=['accuracy',tf.keras.metrics.AUC(name='AUC',multi_label=True)])
#     model.compile(optimizer=sgd, loss=loss1, metrics=['accuracy',tf.keras.metrics.AUC(name='AUC',multi_label=True)])
#     model.compile(optimizer=adam, loss=[focal_loss(gamma=5,alpha=0.75)], metrics=['accuracy'])
    history = model.fit([train_input_signal,train_input_clinical,train_input_wavelet],train_y, batch_size=batch_size, epochs=epochs,
    validation_data=([test_input_signal,test_input_clinical,test_input_wavelet],test_y),verbose=1,shuffle=True,class_weight={0:0.20,1:1.})  

#model test
y_score = model.predict([test_input_signal,test_input_clinical,test_input_wavelet])
scores = model.evaluate([test_input_signal,test_input_clinical,test_input_wavelet],test_y, verbose=0)
print(scores)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
label_base = test_y1.argmax(axis = 1)
predicted = y_score.argmax(axis = 1)
y_score_t = y_score[:,1]
print('AUC: %.4f' % metrics.roc_auc_score(label_base,y_score_t))
print ('ACC: %.4f' % metrics.accuracy_score(label_base,predicted))
print ('Precision: %.4f' %metrics.precision_score(label_base,predicted))
print ('Recall: %.4f' % metrics.recall_score(label_base,predicted))
print ('F1-score: %.4f' %metrics.f1_score(label_base,predicted))
print(metrics.confusion_matrix(label_base,predicted))

precision, recall, _threshold = metrics.precision_recall_curve(label_base,y_score_t)
PR_Area = metrics.auc(recall, precision)
plt.figure(figsize=(6,6))
plt.title('PRC')
plt.plot(recall,precision)
plt.plot([0,1],[1,0],'r--')
plt.xlabel('Recall')# make axis labels
plt.ylabel('Precision')
plt.show()

false_positive_rate,true_positive_rate,thresholds=metrics.roc_curve(label_base,y_score_t)
AUC = metrics.auc(false_positive_rate, true_positive_rate)
plt.figure(figsize=(6,6))
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()