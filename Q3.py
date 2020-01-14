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
    temp_B = np.dot(temp_A, X.T)  # XT·y
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




def Learning_Curve(data, test, λ):
    R = []
    Repeat_Times = 10
    for j in range(1, len(data) + 1):  # 递增Train的data量
        Record_MSE = 0
        for i in range(0, Repeat_Times):  # 每次结果求10次的平均值
            Selection = random.sample(range(0, len(data)), j)# 生成不重复的伪随机数，个数的选区范围在数据的MAX和1之间，至少的有一个数据
            Train_Selected = data[Selection]
            W = get_weight_CV(Train_Selected, λ)
            temp_MSE = get_MSE_CV(test, W)
            Record_MSE += temp_MSE
        MSE = Record_MSE/Repeat_Times  # 得到最终的MSE
        R.append(MSE)
        #print(R)
    plt.plot(range(1, len(data)+1), R, color='pink', label='Learning Curve')
    plt.xlabel('Amount of Training instances')
    plt.ylabel('MSE')
    plt.title('Learning Curve')
    plt.show()


if __name__ == "__main__":

    '''Global ariable'''
    λ = []
    for i in range(0, 151):
        λ.append(i)
    train_sets = [D1_train, D2_train, D3_train, D3_train_50, D3_train_100, D3_train_150]
    test_sets = [D1_test, D2_test, D3_test]


    # Question 3
for λ_test in [1,25,150]:
    Learning_Curve(D3_train, D3_test, λ_test)

