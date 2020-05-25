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
def prior_distri(w0,w1,w2):
    # a = 1 / pow(pow(2*math.pi*pow(10,-6),3),0.5)
    alpha = pow(10,-6)
    a = pow(pow(alpha/(2*math.pi),0.5),3)
    weight = pow(w0,2) + pow(w1,2) + pow(w2,2)
    b = math.exp((-alpha)*weight/2)
    return a*b
def likelihood_distri(w0,w1,w2,x,t):
    a = 1/(2*math.pi)
    y = w0*base(x,0) + w1*base(x,1) + w2*base(x,2)
    b = math.exp(-((t-y)**2)/2)
    return a*b
def max_min_normalization(max_x,min_x,x):
    return (x-min_x)/(max_x-min_x)

def plot_mesh(x,y,data,fig_name,save_path):
    plt.figure(fig_name,figsize=(10,5))
    plt.xlabel('w1', fontsize = 16)
    plt.ylabel('w2', fontsize = 16)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    plt.pcolormesh(x, y, data, cmap='Spectral_r')
    # plt.colorbar()
    plt.savefig(save_path)
    plt.show()
    # plt.close()
    # plt.clf()
def line_y(x,weight):
    line = np.linspace(0,1,len(x))
    for i in range(0,len(x)):
        line[i] = weight[0]*base(x[i],0) + weight[1]*base(x[i],1) + weight[2]*base(x[i],2)
    return line

def plot_line(x,line_weight,train_x,train_t,fig_name,save_path):
    plt.figure(fig_name,figsize=(10,5))
    plt.xlabel('x', fontsize = 16)                        # 設定坐標軸標籤
    plt.ylabel('t', fontsize = 16)                        # 設定坐標軸標籤
    plt.xticks(fontsize = 10)                                 # 設定坐標軸數字格式
    plt.yticks(fontsize = 10)

    draw_y = []
    for i in range(0,5):
        draw_y.append(line_y(x,line_weight[i]))

    line1, = plt.plot(x, draw_y[0], color = 'pink', linewidth = 1, label = 'y(x,w)')
    line2, = plt.plot(x, draw_y[1], color = 'pink', linewidth = 1, label = 'y(x,w)')
    line3, = plt.plot(x, draw_y[2], color = 'pink', linewidth = 1, label = 'y(x,w)')
    line4, = plt.plot(x, draw_y[3], color = 'pink', linewidth = 1, label = 'y(x,w)')
    line5, = plt.plot(x, draw_y[4], color = 'pink', linewidth = 1, label = 'y(x,w)')

    sca = plt.scatter(train_x,train_t,s=15,c='gray',marker='o',edgecolor='black',label='input data')
    plt.legend(handles = [line1, sca], loc='upper right')

    plt.savefig(save_path)
    plt.clf()
    # plt.cla()

data_size = [5,10,30,80]

inputfile = osp.join('..','dataset','1_data.mat')
mat = scipy.io.loadmat(inputfile)

input_x = []  #100 data
input_t = []  #100 data
for i in range(0,100):
    input_t.append(float(mat['t'][i]))
    input_x.append(float(mat['x'][i]))

# index = 3
# train_size = data_size[index]
# phi = np.zeros((train_size,3))
train_t = []
train_x = []



mesh_size = 100
w1 = np.linspace(-5, 5, mesh_size)
w2 = np.linspace(-5, 5, mesh_size)
w3 = np.linspace(-5, 5, mesh_size)

W = np.zeros((5,3)) # 5 lines' w0 w1 w2

# initial prior
prior = np.zeros((mesh_size,mesh_size,mesh_size))
max_prior = prior_distri(0,0,0)
min_prior = prior_distri(0,0,0)
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        for k in range(0,mesh_size):
            prior[i][j][k] = prior_distri(w1[j],w2[i],w3[k])
            if prior[i][j][k]>max_prior:
                max_prior = prior[i][j][k]
            elif prior[i][j][k]<min_prior:
                min_prior = prior[i][j][k]

for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        for k in range(0,mesh_size):
            prior[i][j][k] = max_min_normalization(max_prior,min_prior,prior[i][j][k])

# initial likelihood (using train data 1)
likelihood = np.zeros((mesh_size,mesh_size,mesh_size))
max_likeli = likelihood_distri(0,0,0,input_x[0],input_t[0])
min_likeli = likelihood_distri(0,0,0,input_x[0],input_t[0])
print('read data 1')
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        for k in range(0,mesh_size):
            likelihood[i][j][k] = likelihood_distri(w1[j],w2[i],w3[k],input_x[0],input_t[0])
            if max_likeli < likelihood[i][j][k] :
                max_likeli = likelihood[i][j][k]
            elif min_likeli > likelihood[i][j][k]:
                min_likeli = likelihood[i][j][k]
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        for k in range(0,mesh_size):
            likelihood[i][j][k] = max_min_normalization(max_likeli,min_likeli,likelihood[i][j][k])

# posterior
posterior = np.zeros((mesh_size,mesh_size,mesh_size))
max_posterior = likelihood[0][0][0] * prior[0][0][0]
min_posterior = likelihood[0][0][0] * prior[0][0][0]
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        for k in range(0,mesh_size):
            posterior[i][j][k] = likelihood[i][j][k] * prior[i][j][k]
            if posterior[i][j][k] > max_posterior:
                max_posterior = posterior[i][j][k]
            elif posterior[i][j][k] < min_posterior:
                min_posterior = posterior[i][j][k]
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        for k in range(0,mesh_size):
            posterior[i][j][k] = max_min_normalization(max_posterior,min_posterior,posterior[i][j][k])

draw_x = np.linspace(0,2,100)
train_t.append(input_t[0])
train_x.append(input_x[0])
# update data
for index in range(1,data_size[3]):
    w_index = 0
    train_t.append(input_t[index])
    train_x.append(input_x[index])
    print('read data {:d}'.format(index+1))
    # likelihood update
    max_likeli = likelihood_distri(0,0,0,input_x[index],input_t[index])
    min_likeli = likelihood_distri(0,0,0,input_x[index],input_t[index])
    for i in range(0,mesh_size):
        for j in range(0,mesh_size):
            for k in range(0,mesh_size):
                likelihood[i][j][k] = likelihood_distri(w1[j],w2[i],w3[k],input_x[index],input_t[index])
                if max_likeli < likelihood[i][j][k] :
                    max_likeli = likelihood[i][j][k]
                elif min_likeli > likelihood[i][j][k]:
                    min_likeli = likelihood[i][j][k]
    for i in range(0,mesh_size):
        for j in range(0,mesh_size):
            for k in range(0,mesh_size):
                likelihood[i][j][k] = max_min_normalization(max_likeli,min_likeli,likelihood[i][j][k])
    # posterior/prior update
    for i in range(0,mesh_size):
        for j in range(0,mesh_size):
            for k in range(0,mesh_size):
                posterior[i][j][k] = likelihood[i][j][k] * posterior[i][j][k]
                if (posterior[i][j][k] > 0.8) and (w_index < 5):
                    W[w_index][0] = w1[j]
                    W[w_index][1] = w2[i]
                    W[w_index][2] = w3[k]
                    w_index = w_index + 1
    if index+1 in data_size:
        file_name = 'N = '+str(index+1)+' y(x,w)'
        save_path = osp.join('..','output','1-1',file_name)
        plot_line(draw_x,W,train_x,train_t,file_name,save_path)


