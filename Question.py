#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:11:52 2019

@author: zixingzhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df_test = pd.read_csv("./test-1000-100.csv")
df_train = pd.read_csv("./train-1000-100.csv")


def get_weight_CV(train, λ):
    X = train[:, 0:-1]
    Y = train[:, -1]
    b = np.ones(train.shape[0])
    X = np.insert(X, 0, b, axis=1)
    temp_A = np.linalg.inv(np.dot(X.T, X) + (λ * np.identity(np.dot(X.T, X).shape[0])))  # (XT·X+λ·I)^-1
    temp_B = np.dot(temp_A, X.T)  # XT·y
    weight = (np.dot(temp_B, Y))  # (XT·X+λ·I)^-1 XT·y  得到所有λ下能够得到的权重
    return weight
def get_MSE_CV(test, weight):
    X = test[:, 0:-1]
    Y = test[:, -1]
    b = np.array(np.ones(test.shape[0]))
    X = np.insert(X, 0, b, axis=1)
    for i in range(0, len(Y)):
        XW = np.dot(X, weight)  # XW
        MSE = ((np.linalg.norm(np.mat(XW - Y)) ** 2) / X.shape[0])  # ||XW-y||^2/N
    return MSE


def get_MSE(train, test, λ):
    X = train[:, 0:-1]
    Y = train[:, -1]
    b = np.ones(train.shape[0])
    X = np.insert(X, 0, b, axis=1)
    temp_A = np.linalg.inv(np.dot(X.T, X) + (λ * np.identity(np.dot(X.T, X).shape[0])))  # (XT·X+λ·I)^-1
    temp_B = np.dot(temp_A, X.T)  # XT·y
    weight = (np.dot(temp_B, Y))  # (XT·X+λ·I)^-1 XT·y  得到所有λ下能够得到的权重
    X = test[:, 0:-1]
    Y = test[:, -1]
    b = np.array(np.ones(test.shape[0]))
    X = np.insert(X, 0, b, axis=1)
    for i in range(0, len(Y)):
        XW = np.dot(X, weight)  # XW
        MSE = ((np.linalg.norm(np.mat(XW - Y)) ** 2) / X.shape[0])  # ||XW-y||^2/N
    return MSE
    d2_row = data_train.shape[0]
    Train_Column = data_train.shape[1]
    c2 = data_train[:, :Train_Column - 1]
    r2 = np.ones(d2_row)
    Train_y = data_train[:,- 1]
    Test_y = data_test[:, - 1]
    Train_X = np.column_stack((r2, c2))  # train
    # lambda2_axis, mse2_axis = [], []
    a2 = np.dot(Train_X.T, Train_X)  #XT`X
    b2 = np.linalg.inv(a2 + λ * np.identity(Train_Column)) #XT`X+l`I
    v2 = np.dot(b2, Train_X.T) #XT`X+l`I`XT
    #print(v2.shape)
    W = np.dot(v2, Train_y)
    m2 = np.dot(Train_X, W) - Test_y #XW-y
    n2 = (np.linalg.norm(m2)) ** 2  # ||XW-y||^2
    MSE = n2 / len(data_test)  # ||XW-y||^2/N
    return MSE


def Learning_Curve(data, test, λ):
    R = []
    Repeat_Times = 10
    for j in range(1, len(data) + 1):  # 递增Train的data量
        Record_MSE = 0
        for i in range(0, Repeat_Times):  # 每次结果求10次的平均值
            Selection = random.sample(range(0, len(data)), j)# 生成不重复的伪随机数，个数的选区范围在数据的MAX和1之间，至少的有一个数据
            Train_Selected = data[Selection]
            temp_MSE = get_MSE(Train_Selected,test,λ)
            #W = get_weight_CV(Train_Selected, λ)
            #temp_MSE = get_MSE_CV(test, W)
            Record_MSE += temp_MSE
        MSE = Record_MSE/Repeat_Times  # 得到最终的MSE
        R.append(MSE)
        #print(R)
    plt.plot(range(1, len(data)+1), R, color='pink', label='Learning Curve')
    plt.xlabel("size")
    plt.ylabel("mse with lambda = " + str(r))
    plt.title('Q3')
    plt.show()

data_test = np.array(df_test[:], dtype=np.float)
data_train = np.array(df_train[:], dtype=np.float)


if __name__=='__main__':
    print(get_MSE(data_train,data_test,1))
    for λ in [1,25,150]:
        Learning_Curve(data_train, data_test,λ)

#print(get_MSE(data_train, data_test, 1))
'''
    for j in range(len(size)):
        mse_axis = []
        for i in range(10):
            get_MSE(data_train,data_test,λ)


            d2_row = data_train.shape[0]
            Train_Column = data_train.shape[1]
            c2 = data_train[:, :Train_Column - 1]
            r2 = np.ones(d2_row)
            Train_y = data_test[:, Train_Column - 1]
            Train_X = np.column_stack((r2, c2))  # train
            lambda2_axis, mse2_axis = [], []
            a2 = np.dot(Train_X.T, Train_X)  #
            b2 = np.linalg.inv(a2 + r * np.identity(Train_Column))
            v2 = np.dot(b2, Train_X.T)
            W = np.dot(v2, Train_y)
            m2 = np.dot(Train_X, W) - Train_y
            n2 = (np.linalg.norm(m2)) ** 2  #||XW-y||^2
            mse2_pre = n2 / len(test_data)  #||XW-y||^2/N
            mse2 = np.array(mse2_pre)
            mse2_axis.append(np.min(mse2))
            lambda2_axis.append(r2)

            z = np.random.sample(len(data_test), size[j])  # 随机数列表
            d1_row = data_test.shape[0]
            d1_col = data_test.shape[1]
            c1 = data_test[:, :d1_col - 1]  # 取test x
            r1 = np.ones(d1_row)  # 构造截距列
            y = data_test[:, d1_col - 1]  # 取y
            y_z = y[z]
            x = np.column_stack((r1, c1))  # test_x
            x_z = x[z]
            mse_axis = []
            a = np.dot(x_z.T, x_z)
            b = np.linalg.inv(a + r * np.identity(d1_col))
            v = np.dot(b, x_z.T)
            w = np.dot(v, y_z)
            m = np.dot(x, w) - y
            n = (np.linalg.norm(m)) ** 2
            mse_pre = n / d1_row
            mse = np.array(mse_pre)
            mse_axis.append(np.min(mse))


        mse_test_array[j] = np.average(mse_axis)
    plt.plot(size, mse_test_array, label="MSE_Test")
    plt.xlabel("size")
    plt.ylabel("mse with lambda = " + str(r))
    plt.legend()
    plt.show()
'''