import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path as osp
import scipy.io
import math

def sigmoid(x):
    return 1/(1+math.exp(-x))
def base(x,j):      # j = 0 , 1 , 2
    mu = j * 2 / 3
    s = 0.1
    return sigmoid((x-mu)/s)

def mean_variance(x,train_t,phi):
    base_x = np.zeros((1,3))
    for i in range(0,3):
        base_x[0][i] = base(x,i)
    A = np.identity(3)
    Sn_inv = pow(10,-6) * A + (phi.T).dot(phi)
    B = base_x.dot(np.linalg.inv(Sn_inv))
    C = B.dot(phi.T)
    m_x = C.dot(train_t)
    var_x = 1 + B.dot(base_x.T)

    return m_x , var_x
def posterior_distri(phi,train_t):
    A = np.identity(3)
    Sn_inv = pow(10,-6) * A + (phi.T).dot(phi)
    B = (np.linalg.inv(Sn_inv)).dot(phi.T)
    mean = B.dot(train_t)
    return mean

data_size = [5,10,30,80]

inputfile = osp.join('..','dataset','1_data.mat')
mat = scipy.io.loadmat(inputfile)

input_x = []  #100 data
input_t = []  #100 data
for i in range(0,100):
    input_t.append(float(mat['t'][i]))
    input_x.append(float(mat['x'][i]))

for index in range(0,4):
    train_size = data_size[index]
    phi = np.zeros((train_size,3))
    train_t = np.zeros((train_size,1))
    train_x = np.zeros((train_size,1))
    for i in range(0,train_size):
        for j in range(0,3):
            phi[i][j] = base(input_x[i],j)
        train_t[i][0] = input_t[i]
        train_x[i][0] = input_x[i]

    # print(phi)

    A = (phi.T).dot(phi)
    B = np.linalg.inv(A).dot(phi.T)
    W = B.dot(train_t)
    # print(W)
    W_mean = posterior_distri(phi,train_t)
    # print(W_mean)

    draw_x = np.linspace(0,2,100)
    draw_mean = np.linspace(0,1,100)
    draw_var = np.linspace(0,1,100)
    draw_upper = np.linspace(0,1,100)
    draw_lowwer = np.linspace(0,1,100)
    for i in range(0,100):
        draw_mean[i] , draw_var[i] = mean_variance(draw_x[i],train_t,phi)
        draw_upper[i] = draw_mean[i] + draw_var[i]
        draw_lowwer[i] = draw_mean[i] - draw_var[i]
    fig_name = 'N = '+str(train_size)
    plt.figure(fig_name,figsize=(10,5))
    plt.xlabel('x', fontsize = 16)                        # 設定坐標軸標籤
    plt.ylabel('t', fontsize = 16)                        # 設定坐標軸標籤
    plt.xticks(fontsize = 10)                                 # 設定坐標軸數字格式
    plt.yticks(fontsize = 10)
    # plt.grid(color = 'red', linestyle = '--', linewidth = 1)  # 設定格線顏色、種類、寬度
    # plt.ylim(-1, 5)                                          # 設定y軸繪圖範圍
    # 繪圖並設定線條顏色、寬度、圖例
    line1, = plt.plot(draw_x, draw_mean, color = 'red', linewidth = 2, label = 'Mean')             

    area = plt.fill_between(draw_x, draw_lowwer,draw_upper,facecolor='pink', interpolate=True, label = 'Variance')
    sca = plt.scatter(train_x,train_t,s=15,c='gray',marker='o',edgecolor='black',label='input data')
    plt.legend(handles = [line1, area, sca], loc='upper right')
    save_fig_path = osp.join('..','output','1-2',fig_name)
    # plt.savefig(save_fig_path)
    plt.show()