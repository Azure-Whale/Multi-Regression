import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

'''Import all sets from local'''
D1_test = pd.read_csv('./test-100-10.csv')
D1_train = pd.read_csv('./train-100-10.csv')
D2_test = pd.read_csv('./test-100-100.csv')
D2_train = pd.read_csv('./train-100-100.csv')
D3_test = pd.read_csv('./test-1000-100.csv')
D3_train_50 = pd.read_csv('./train-50(1000)-100.csv')
D3_train_100 = pd.read_csv('./train-100(1000)-100.csv.csv')
D3_train_150 = pd.read_csv('./train-150(1000)-100.csv')
D3_train = pd.read_csv('./train-1000-100.csv')
'''Transfer to array'''
D1_test = np.array(D1_test)
D1_train = np.array(D1_train)
D2_test = np.array(D2_test)
D2_train = np.array(D2_train)
D3_test = np.array(D3_test)
D3_train = np.array(D3_train)
D3_train_50 = np.array(D3_train_50)
D3_train_100 = np.array(D3_train_100)
D3_train_150 = np.array(D3_train_150)


def get_weight_CV(train, λ):
    X = train[:, 0:-1]
    Y = train[:, -1]
    b = np.ones(train.shape[0])
    X = np.insert(X, 0, b, axis=1)
    temp_A = np.linalg.inv(np.dot(X.T, X) + (λ * np.identity(np.dot(X.T, X).shape[0])))  # (XT·X+λ·I)^-1
    temp_B = np.dot(temp_A, X.T)  # (XT·X+λ·I)^-1 XT
    weight = (np.dot(temp_B, Y))  # (XT·X+λ·I)^-1 XT·y  得到所有λ下能够得到的权重
    return weight


#  Error Measure
def get_MSE_CV(test, weight):
    X = test[:, 0:-1]
    Y = test[:, -1]
    b = np.array(np.ones(test.shape[0]))
    X = np.insert(X, 0, b, axis=1)
    for i in range(0, len(Y)):
        XW = np.dot(X, weight)  # XW
        MSE = ((np.linalg.norm(np.mat(XW - Y)) ** 2) / X.shape[0])  # ||XW-y||^2/N
    return MSE



def Get_Best(data, λ):
    rec_MSE = []
    num = 0
    fold = 10  # 所要分的层数
    for i in λ:
        Sum_MSE = 0
        # 计算当前λ的MSE
        for j in range(0, len(data), int(len(data) / fold)):  # 分层数为10，把数据10等分
            Train_set = np.row_stack(
                (data[0:j], data[(j + int(len(data) / fold)):len(data)]))  # Select Train set (except the jth fold)
            Test_set = data[j:j + int(len(data) / fold)]  # select the jth fold
            W_CV = get_weight_CV(Train_set, i)  # 计算当前层的W
            temp_MSE = get_MSE_CV(Test_set, W_CV)  # 计算当前层的MSE
            Sum_MSE += temp_MSE  # 计算所有层MSE的和，完成当前λ的计算
        MSE = Sum_MSE / fold  # 求Average Performance也就是10层的MSE的平均数
        rec_MSE.append(MSE)  # 记录当前λ得到的MSE，一共遍历所有的λ
    for i in λ:
        if rec_MSE[i] == min(rec_MSE):
            num = i
    return [rec_MSE, num]


def CrossValidation(data, λ):
    Result = Get_Best(data, λ)
    return Result

if __name__ == "__main__":

    '''Global ariable'''
    λ = []
    for i in range(0, 151):
        λ.append(i)
    train_sets = [D1_train, D2_train, D3_train, D3_train_50, D3_train_100, D3_train_150]
    test_sets = [D1_test, D2_test, D3_test]


# Question2 using CrossValidation
print("Question 2\nBest MSE and corresponding λ")
print("########################################################")
plt.figure(3)
names = ['Data_set1', "Data_set_2", "Data_set_3", "Data_set_4", "Data_set_5", "Data_set_6"]
iterator = 0

for data in train_sets:
    print([min(CrossValidation(data, λ)[0]), CrossValidation(data, λ)[1]])  # 输出六个数据集对应的MSE和最优λ
print("########################################################")
