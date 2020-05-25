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
def prior_distri(w0,w1):
    # a = 1 / pow(pow(2*math.pi*pow(10,-6),3),0.5)
    alpha = pow(10,-6)
    a = pow(pow(alpha/(2*math.pi),0.5),3)
    weight = pow(w0,2) + pow(w1,2)
    b = math.exp((-alpha)*weight/2)
    return a*b
def likelihood_distri(w0,w1,x,t):
    a = 1/(2*math.pi)
    y = w0*base(x,0) + w1*base(x,1)
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

data_size = [5,10,30,80]

inputfile = osp.join('..','dataset','1_data.mat')
mat = scipy.io.loadmat(inputfile)

input_x = []  #100 data
input_t = []  #100 data
for i in range(0,100):
    input_t.append(float(mat['t'][i]))
    input_x.append(float(mat['x'][i]))

index = 3
train_size = data_size[index]
phi = np.zeros((train_size,3))
train_t = np.zeros((train_size,1))
train_x = np.zeros((train_size,1))
for i in range(0,train_size):
    for j in range(0,3):
        phi[i][j] = base(input_x[i],j)
    train_t[i][0] = input_t[i]
    train_x[i][0] = input_x[i]



mesh_size = 500
w1 = np.linspace(-10, 10, mesh_size)
w2 = np.linspace(-10, 10, mesh_size)

# initial prior
prior = np.zeros((mesh_size,mesh_size))
max_prior = prior_distri(0,0)
min_prior = prior_distri(0,0)
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        prior[i][j] = prior_distri(w1[j],w2[i])
        if prior[i][j]>max_prior:
            max_prior = prior[i][j]
        elif prior[i][j]<min_prior:
            min_prior = prior[i][j]

for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        prior[i][j] = max_min_normalization(max_prior,min_prior,prior[i][j])

# initial likelihood (using train data 1)
likelihood = np.zeros((mesh_size,mesh_size))
max_likeli = likelihood_distri(0,0,input_x[0],input_t[0])
min_likeli = likelihood_distri(0,0,input_x[0],input_t[0])
print('read data 1')
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        likelihood[i][j] = likelihood_distri(w1[j],w2[i],input_x[0],input_t[0])
        if max_likeli < likelihood[i][j] :
            max_likeli = likelihood[i][j]
        elif min_likeli > likelihood[i][j]:
            min_likeli = likelihood[i][j]
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        likelihood[i][j] = max_min_normalization(max_likeli,min_likeli,likelihood[i][j])

# caluculate posterior fig
posterior = np.zeros((mesh_size,mesh_size))
max_posterior = likelihood[0][0] * prior[0][0]
min_posterior = likelihood[0][0] * prior[0][0]
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        posterior[i][j] = likelihood[i][j] * prior[i][j]
        if posterior[i][j] > max_posterior:
            max_posterior = posterior[i][j]
        elif posterior[i][j] < min_posterior:
            min_posterior = posterior[i][j]
for i in range(0,mesh_size):
    for j in range(0,mesh_size):
        posterior[i][j] = max_min_normalization(max_posterior,min_posterior,posterior[i][j])

# update data
for index in range(1,data_size[3]):
    print('read data {:d}'.format(index+1))
    # likelihood update
    max_likeli = likelihood_distri(0,0,input_x[index],input_t[index])
    min_likeli = likelihood_distri(0,0,input_x[index],input_t[index])
    for i in range(0,mesh_size):
        for j in range(0,mesh_size):
            likelihood[i][j] = likelihood_distri(w1[j],w2[i],input_x[index],input_t[index])
            if max_likeli < likelihood[i][j] :
                max_likeli = likelihood[i][j]
            elif min_likeli > likelihood[i][j]:
                min_likeli = likelihood[i][j]
    for i in range(0,mesh_size):
        for j in range(0,mesh_size):
            likelihood[i][j] = max_min_normalization(max_likeli,min_likeli,likelihood[i][j])
    # posterior/prior update
    for i in range(0,mesh_size):
        for j in range(0,mesh_size):
            posterior[i][j] = likelihood[i][j] * posterior[i][j]
    
    if index+1 in data_size:
        file_name = 'N = '+str(index+1)+' prior'
        save_path = osp.join('..','output','1-3',file_name)
        plot_mesh(w1,w2,posterior,file_name,save_path)


