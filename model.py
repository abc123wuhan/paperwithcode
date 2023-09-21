#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tensorflow.keras.models import Model,load_model
import keras as K
from keras.layers.core import Dense
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras import backend
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional

def models(input_dim = 225):
# 指定显卡
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # 模型
    model = K.models.Sequential()
    feature_size = omni2_num * (history_length - 1) + swarm_nums
    model.add(K.layers.Dense(units=64, input_dim=input_dim, kernel_initializer = 'normal', activation='sigmoid'))
    # 对于TensorFlow不能表达的复杂激活函数，如含有可学习参数的激活函数，可通过高级激活函数实现，可以在 keras.layers.advanced_activations 模块中找到。
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.BatchNormalization())

    model.add(K.layers.Dense(units=32, kernel_initializer = 'normal', activation='sigmoid'))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.BatchNormalization())

    model.add(K.layers.Dense(units=16, kernel_initializer = 'normal', activation='sigmoid'))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.BatchNormalization())

    model.add(K.layers.Dense(units=16, kernel_initializer = 'normal', activation='sigmoid'))
    model.add(K.layers.Dropout(0.2))
    model.add(K.layers.BatchNormalization())

    model.add(K.layers.Dense(units=1))
    model.summary()

    # model.compile(loss='mse', optimizer='Adam''Adamax' ,metrics=['mae'])   #loss:hinge
    model.compile(loss=rmse,  optimizer='Adamax',metrics=['mse'])
    return model

def generate_arrays_from_file(train_index):  

    while 1:
        for j in range(batch_step_nums):  

            cur_idx = train_index[j * batch_size: (j + 1) * batch_size]
            cur_feature = np.concatenate([feature[cur_idx - k, 9:] for k in range(1, history_length)], axis=-1)
            cur_feature = np.concatenate([cur_feature, feature[cur_idx, :9]], axis=-1)
            cur_dens = dens[cur_idx]
            yield (cur_feature, cur_dens)

length = len(train_feature) // 10
parts = []

for i in range(10):
    parts.append(train_feature.index[i * length: length + i * length])
    if i == 9:
        parts[-1] = train_feature.index[i * length: ]

kf = KFold(10, shuffle=False, random_state=None)
A = []
B = []
Q = []
history_length = 5
swarm_nums = 9
omni2_num = 52
ensembel = 0

for train_idx in parts:
    train_index, test_index = train_idx[:int(len(train_idx) * 0.9)], train_idx[int(len(train_idx) * 0.9):]
    Q.append(test_index)

    model = models()
    # model.compile(loss=hinge,  optimizer='Nadam' ,metrics=['mae'])
    model.compile(loss=rmse,optimizer='Adamax', metrics=['mse'])
    filepath = f'0114goce_best_model_ensembel_{ensembel}.h5'     
    #filepath = 'weights.best.h5'
    checkpoint = ModelCheckpoint(filepath,  monitor='val_mse', verbose=0, save_best_only=True, mode='min', period=1) # 决定性能最佳模型的评判准则
    callbacks_list = [checkpoint]

    batch_size = 128         
    batch_step_nums = len(train_index) // batch_size      

    valid_ind = np.random.permutation(idx[int(n_data * .8):int(n_data * .9)])
    valid_feature = np.concatenate([feature[valid_ind - k, 9:] for k in range(1, history_length)], axis=-1)
    # 最后一个数据
    valid_feature = np.concatenate([valid_feature, feature[valid_ind, :9]], axis=-1)
    valid_dens = dens[valid_ind]
    if trainenable:    # model.fit_generator #model.fit
        history=model.fit_generator(generate_arrays_from_file(train_index),
                            steps_per_epoch=batch_step_nums,
                            validation_data=(valid_feature, valid_dens),
                            epochs=20,
                            max_queue_size=1,
                            workers=1,
                            callbacks=callbacks_list)

    test_feature = np.concatenate([feature[test_index - k, 9:] for k in range(1, history_length)], axis=-1)
    test_feature = np.concatenate([test_feature, feature[test_index, :9]], axis=-1)
    test_dens = dens[test_index]
    model = load_model(f'0114goce_best_model_ensembel_{ensembel}.h5',custom_objects={'rmse': rmse})
    y_predict = model.predict(test_feature)
    pr = model.predict(test_feature)
    A.append(pr)

    test_feature = np.concatenate([feature[test_ind - k, 9:] for k in range(1, history_length)], axis=-1)
    test_feature = np.concatenate([test_feature, feature[test_ind, :9]], axis=-1)
    pr = model.predict(test_feature)
    B.append(pr)
    ensembel += 1

AC = np.vstack(A)
BC = np.hstack(B)
BC = BC.mean(axis=1)
QI = np.hstack(Q)

model = keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=[None, 1])),
                                 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
                                 keras.layers.Dense(1)])

model.compile(loss=rmse,
              optimizer='Adamax', metrics=['mse'])

filepath = '0114goce_best_model_ensembel_final.h5'
checkpoint = ModelCheckpoint(filepath,  monitor='val_mse', verbose=1, save_best_only=True, mode='min', period=1)
callbacks_list = [checkpoint]
if trainenable:
    history1=model.fit(AC.reshape(-1, 1, 1),dens[QI],
              epochs=20,
              batch_size = 128,
              verbose=1,
              validation_split = 0.2,
              callbacks=callbacks_list)



