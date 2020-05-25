import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path as osp
import scipy.io
import math

train_size = 60
beta = 1

linear_kernel = [0,0,0,1]
squared_exp_kernel = [1,4,0,0]
exp_quadratic1_kernel = [1,4,0,5]
exp_quadratic2_kernel = [1,32,5,5]

def read_file(path):
    mat = scipy.io.loadmat(path)
    input_x = []  #100 data
    input_t = []  #100 data
    for i in range(0,100):
        input_t.append(float(mat['t'][i]))
        input_x.append(float(mat['x'][i]))
    
    return input_x,input_t

def kernel(Xn,Xm,kernel_theta):
    return kernel_theta[0]*math.exp(-0.5*kernel_theta[1]*np.dot(Xn-Xm,Xn-Xm))+kernel_theta[2]+kernel_theta[3]*np.dot(Xn.T,Xm)

def covariance_matrix(train_data,kernel_theta):
    C = np.zeros((len(train_data),len(train_data)))
    for i in range(len(train_data)):
        for j in range(len(train_data)):
            if i==j:
                C[i][j] = kernel(train_data[i][1],train_data[j][1],kernel_theta) + beta
            else:
                C[i][j] = kernel(train_data[i][1],train_data[j][1],kernel_theta)
    return C

def predict_mean_var(train_data,test_x,kernel_theta):
    K = np.zeros((len(train_data),1))
    C = covariance_matrix(train_data,kernel_theta)
    t = np.zeros((len(train_data),1))

    for i in range(len(train_data)):
        K[i][0] = kernel(train_data[i][1],test_x,kernel_theta)
        t[i][0] = train_data[i][0]

    C_inv = np.linalg.inv(C)
    A = np.dot(K.T,C_inv)
    mean = np.dot(A,t)
    var = kernel(test_x,test_x,kernel_theta) + beta - np.dot(A,K)

    return mean , var

def RMS(train_data,test_data,kernel_theta):
    sum = 0
    for i in range(len(train_data)):
        mean_x,var_x = predict_mean_var(train_data,train_data[i][1],kernel_theta)
        sum += (mean_x-train_data[i][0])**2
    rms_train = (sum / len(train_data))**0.5
    sum = 0
    for j in range(len(test_data)):
        mean_x,var_x = predict_mean_var(train_data,test_data[j][1],kernel_theta)
        sum += (mean_x-test_data[j][0])**2
    rms_test = (sum / len(test_data))**0.5

    return float(rms_train) , float(rms_test)

def plot_mean_var(train_data,kernel_theta,fig_name,train_rms,test_rms):
    draw_x = np.linspace(0,2,100)
    draw_mean = np.linspace(0,1,100)
    draw_var = np.linspace(0,1,100)
    draw_upper = np.linspace(0,1,100)
    draw_lowwer = np.linspace(0,1,100)

    for i in range(len(draw_x)):
        draw_mean[i] , draw_var[i] = predict_mean_var(train_data,draw_x[i],kernel_theta)
        draw_upper[i] = draw_mean[i] + draw_var[i]
        draw_lowwer[i] = draw_mean[i] - draw_var[i]
    
    train_x = []
    train_t = []
    for i in range(len(train_data)):
        train_t.append(train_data[i][0])
        train_x.append(train_data[i][1])

    plt.figure(fig_name,figsize=(10,5))
    plt.suptitle(fig_name,fontsize=18)
    plt.title('θ=['+str(kernel_theta[0])+', '+str(kernel_theta[1])+', '+str(kernel_theta[2])+', '+str(kernel_theta[3])+']',fontsize=14)
    plt.xlabel('x',fontsize=14)
    plt.ylabel('t',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    line, = plt.plot(draw_x, draw_mean, color='red', linewidth = 2, label = 'Mean')
    area = plt.fill_between(draw_x, draw_lowwer,draw_upper,facecolor='pink', interpolate=True, label = 'Variance')
    sca = plt.scatter(train_x,train_t,s=15,c='gray',marker='o',edgecolor='black',label='train data')
    plt.legend(handles = [line, area, sca], loc='upper right')
    # plt.text(1.8,5,'Train RMS:'+str(train_rms))
    # plt.text(1.8,4.5,'Train RMS:'+str(test_rms))
    save_fig_path = osp.join('..','output','1',fig_name)
    plt.savefig(save_fig_path)
    plt.show()

if __name__ == '__main__':
    inputfile = osp.join('..','gp.mat')
    input_x,input_t = read_file(inputfile)

    train_data = np.zeros((train_size,2))    # train_data = (t,x)*train_size
    for i in range(train_size):
        train_data[i][0] = input_t[i]
        train_data[i][1] = input_x[i]

    test_data = np.zeros((len(input_x)-train_size,2))   # test_data = (t,x)*(input_size-train_size)
    for i in range(len(input_x)-train_size):
        test_data[i][0] = input_t[i+train_size]
        test_data[i][1] = input_x[i+train_size]

    # 4 kernel:img+RMS
    rms_train , rms_test = RMS(train_data,test_data,linear_kernel)
    plot_mean_var(train_data,linear_kernel,'Linear Kernel',rms_train,rms_test)
    print('Linear Kernel RMS')
    print('\tTrain data:{:0.3f} , Test data:{:0.3f} \n'.format(rms_train,rms_test))
    
    rms_train , rms_test = RMS(train_data,test_data,squared_exp_kernel)
    plot_mean_var(train_data,squared_exp_kernel,'Squared Exp Kernel',rms_train,rms_test)
    print('Squared Exp Kernel RMS')
    print('\tTrain data:{:0.3f} , Test data:{:0.3f} \n'.format(rms_train,rms_test))

    rms_train , rms_test = RMS(train_data,test_data,exp_quadratic1_kernel)
    plot_mean_var(train_data,exp_quadratic1_kernel,'Exp Quadratic Kernel 1',rms_train,rms_test)
    print('Exp Quadratic Kernel-1 RMS')
    print('\tTrain data:{:0.3f} , Test data:{:0.3f} \n'.format(rms_train,rms_test))

    rms_train , rms_test = RMS(train_data,test_data,exp_quadratic2_kernel)
    plot_mean_var(train_data,exp_quadratic2_kernel,'Exp Quadratic Kernel 2',rms_train,rms_test)
    print('Exp Quadratic Kernel-2 RMS')
    print('\tTrain data:{:0.3f} , Test data:{:0.3f} \n'.format(rms_train,rms_test))

    # tune θ (with ARD)