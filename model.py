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
from keras.losses import mean_squared_error
from keras import backend
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Bidirectional
# import time
# start=time.clock()

trainenable = 0
day_all = 1
data = pd.read_csv("H:\code\swarm_extended_omni_data.csv", header=None)
data = data.get_chunk()
# 查看数据详情
#data

glon = data[4]
glat = data[5]
height = data[3]
doy = data[15]
Kp = data[52]
DST = data[54]
ap = data[63]
F107 = data[64]

dens = data[8]
dens = dens.astype("float")
# 删除数据中没有用到的列
df_feature = data.drop([ 0, 1, 2, 8, 14, 16], axis=1)
df_feature = df_feature.astype("float")
data_inf = np.isfinite(df_feature).all(axis=1).squeeze()
# 逐元素测试有限性(非无穷大，不是非数字) np.isfinite()数据为dataframe类型，.all(axis=1)为一列索引一列true，squeeze()从数组的形状中把shape中为1的维度去掉
df_feature = df_feature.loc[data_inf,:]                        # 特征中 过滤 有无效数据的行
# loc[]：通过标签或布尔数组访问一组行和列，如果某个位置的布尔值是True，则选定该row
dens = dens[data_inf]
dens_inf = np.isfinite(dens).squeeze()
df_feature = df_feature.loc[dens_inf,:]
dens = dens[dens_inf]
# 最后再看一下数据情况
df_feature = df_feature.reset_index(drop=True)
#display(df_feature)
# df_feature.shape

idx = np.array(df_feature.index)
# 划分训练集、验证集、测试集
n_data = len(df_feature)
train_ind = np.random.permutation(idx[:int(n_data * .8)])                    # 前80% 作为训练集，并取随机排列
valid_ind = np.random.permutation(idx[int(n_data * .8):int(n_data * .9)])    # 80%-90% 作为验证集，并取随机排列
test_ind = idx[int(n_data * .9):]                                            # 90% -100% 作为测试集，并取随机排列
# len(test_ind)

# 查看训练、测试数据的日期范围                                                     #10000
train_ind_ = idx[:int(n_data * .8)]                   # 前80% 作为训练集，并取随机排列
valid_ind_ = idx[int(n_data * .8):int(n_data * .9)]    # 80%-90% 作为验证集，并取随机排列
test_ind_ = idx[int(n_data * .9):]
a=data[0]
print('训练数据：',a[train_ind_[0]],'-',a[train_ind_[-1]])
print('验证数据：',a[valid_ind_[0]],'-',a[valid_ind_[-1]])
print('测试数据：',a[test_ind_[0]],'-',a[test_ind_[-1]])

if day_all:
    start_time = '2015-02-02'
    end_time = "2015-02-30"

    # start_time = "2014-02-01" # "2014-2-11"  "2010-04-19"
    # end_time = "2014-02-03"   # "2014-3-11"  "2010-04-29"

    test_ind = data[(data[0]>=start_time) & (data[0]<=end_time)].index
    # test_ind = data[data[0]==start_time].index

else:
    print('error')
# print(test_ind[0],test_ind[-1],test_ind.shape)

scaler_data = StandardScaler() # 使用标准化数据集，公式是X_scaled = (X - X.mean()) / X.std()
scaler_dens = StandardScaler()

train_feature = df_feature.loc[train_ind]
# train_ind：训练集的序列号

scaler_data.fit(train_feature)
feature = scaler_data.transform(df_feature)

scaler_dens.fit(dens.values.reshape(-1, 1))
dens = scaler_dens.transform(dens.values.reshape(-1, 1))

# 将数据表类型转化为numpy型
feature = np.array(feature)

# 将密度数据维度进行转换，(len(data),) -> (len(data), 1)
dens = np.reshape(dens, (-1, 1))

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

    # model.compile(loss='mse', optimizer='Adam''Adamax' ,metrics=['mae'])   #loss:hinge（收藏夹有其他介绍）
    model.compile(loss=rmse,  optimizer='Adamax',metrics=['mse'])
    return model


def generate_arrays_from_file(train_index):  # train_index训练集的序列号

    while 1:
        for j in range(batch_step_nums):  # batch_step_nums：一次迭代中要喂多少次数据；

            cur_idx = train_index[j * batch_size: (j + 1) * batch_size]
            cur_feature = np.concatenate([feature[cur_idx - k, 9:] for k in range(1, history_length)], axis=-1)
            cur_feature = np.concatenate([cur_feature, feature[cur_idx, :9]], axis=-1)
            cur_dens = dens[cur_idx]
            yield (cur_feature, cur_dens)

length = len(train_feature) // 10
parts = []

# 把分成十份的训练集的索引号全部放到parts中,最终parts是10行length列
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

# 把十组数据按9：1划分训练集和测试集
for train_idx in parts:
    train_index, test_index = train_idx[:int(len(train_idx) * 0.9)], train_idx[int(len(train_idx) * 0.9):]
    Q.append(test_index)

    def rmse(y_true, y_pre):
        return backend.sqrt(mean_squared_error(y_true, y_pre))

    model = models()
    # model.compile(loss=keras.losses.mean_squared_error,  optimizer='Nadam' ,metrics=['mae'])
    model.compile(loss=rmse,optimizer='Adamax', metrics=['mse'])
    filepath = f'0114goce_best_model_ensembel_{ensembel}.h5'     # 生成的文件名
    #filepath = 'weights.best.h5'
    checkpoint = ModelCheckpoint(filepath,  monitor='val_mse', verbose=0, save_best_only=True, mode='min', period=1) # 决定性能最佳模型的评判准则
    callbacks_list = [checkpoint]

    batch_size = 128          # 每次喂给模型多少条数据
    batch_step_nums = len(train_index) // batch_size      # 每个迭代论要喂多少次数据

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

model = keras.models.Sequential([tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1])),
                                 tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
                                 keras.layers.Dense(1)])
def rmse(y_true, y_pre):
    return backend.sqrt(mean_squared_error(y_true, y_pre))
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
              validation_split = 0.1,
              callbacks=callbacks_list)
    # 将训练数据在模型中训练一定次数，返回loss和测量指标

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.rcParams['figure.figsize']=(6, 4) # 图像尺寸大小

loss1=history1.history["loss"]
test_loss1 = history1.history["val_loss"]
# mse1=history1.history['mse']
# test_mse1 = history1.history["val_mse"]

plt.plot(loss1, label='train', color='r')

# 修改坐标轴的刻度间距和刻度范围
x_major_locator=MultipleLocator(2) #间距是2
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.xlim(-0.5,21)

# 修改坐标轴字体及大小
plt.yticks(fontproperties='Times New Roman', size=10)
plt.xticks(fontproperties='Times New Roman', size=10)

# 设置坐标轴标签及字体大小,labelpad坐标轴标签和坐标轴的距离
plt.xlabel('Epoch', fontsize=15, labelpad=1)
plt.ylabel('Loss', fontsize=15, labelpad=1)
#plt.title(' ', fontsize=14)

# 设置坐标刻度值的大小
plt.tick_params(direction='in', width=1, length=2, pad=3)

plt.legend(fontsize=10)
plt.savefig("H:/论文相关/大气密度预测/论文图/rmse_epoch/loss_epoch_test.jpg",bbox_inches='tight',dpi=300,pad_inches=0.0)
plt.show()

# 转换时间
data_time= pd.read_csv("H:/code/swarm_extended_omni_data.csv", header=None)
ymd=data_time[0]
hms=data_time[1]
time1=ymd[-len(test_ind):]
time2=hms[-len(test_ind):]
# g=[]
# for i in time1:
#     i=int(i[0:4])
#     g.append(i)
#
# f=[]
# for i in time1:
#     i=int(i[5:7])
#     f.append(i)
#
j=[]
for i in time1:
    i=int(i[8:10])
    j.append(i)

k=[]
for i in time2:
    i=float(i[0:2])/24+float(i[3:5])/(24*60)+float(i[6:8])/(24*60*60)
    k.append(i)

# l=[]
# for i in time2:
#     i=float(i[0:2])/(24*30)+float(i[3:5])/(24*60*30)+float(i[6:8])/(24*60*60*30)
#     l.append(i)

time=[]
for m,n in zip(j,k):
    l=m+n
    # l=str(l)
    # l=l[0:3]
    # l=float(l)
    time.append(l)

densitypre = scaler_dens.inverse_transform(BC.reshape(-1,1))

densityopt = scaler_dens.inverse_transform(dens[test_ind].reshape(-1,1))
density_pre=densitypre*1000000000000
density_opt=densityopt*1000000000000

#print(density_pre[0:1],'\n',density_opt[:1],'\n',BC.reshape(-1,1)[:1],'\n',dens[test_ind][:1])

plt.rcParams['figure.figsize']=(18.8, 7.2) # 图像尺寸大小
plt.plot(time[:1000],density_opt[:1000], label='Optivision', linewidth = 2,color='b',alpha = 1)
plt.plot(time[:1000],density_pre[:1000], label='Predition',color='r',linewidth = 2,alpha = 1)

# 修改坐标轴字体及大小
plt.yticks(fontproperties='Times New Roman', size=20)
plt.xticks(fontproperties='Times New Roman', size=20)

# 设置坐标轴标签及字体大小,labelpad坐标轴标签和坐标轴的距离
plt.xlabel('November,2019', fontsize=20, labelpad=1)
plt.ylabel('atmosphere density (10^-13 kg/ m^3)', fontsize=20, labelpad=1)
#plt.title(' ', fontsize=14)

# 设置坐标刻度值的大小
plt.tick_params(direction='in', width=1, length=2, pad=3)#修改刻度线线粗细width参数

# 控制图例的形状大小
plt.legend(fontsize=10)
plt.savefig("H:/论文相关/大气密度预测/论文图/density.jpg",bbox_inches='tight',dpi=300,pad_inches=0.0)
plt.show()



