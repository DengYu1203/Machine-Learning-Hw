from math import pow
from math import exp
import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path as osp

input_x = []    #1096 data
input_t = []    #1096 data
with open(osp.join('..','Dataset','dataset_X.csv'),newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        input_x.append(row)
    # print(input_x[1])           #row 1
    csvfile.close
with open(osp.join('..','Dataset','dataset_T.csv'),newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        input_t.append(row)     
    # print(input_t[1])          #row 1
    csvfile.close
with open("hw3-1_Output.txt", "w") as text_file:
    data_max = len(input_x)

    fold = int((data_max-1) / 8)
    N = 1
    print('Fold = 8\n',file=text_file)
    print('Fold = 8\n')
    for N in range(1,9):
        print('(N = {:d})'.format(N),file=text_file)
        print('(N = {:d})'.format(N))
        train_list = []
        valid_list = []
        for i in range(1,fold*(N-1)):
            train_list.append(i)
        for i in range(fold*(N-1),fold*N+1):
            if i == 0:
                i = i + 1
                continue
            valid_list.append(i)
        for i in range(fold*N+1,data_max):
            train_list.append(i)

        # for i in valid_list:
        #     print(i)
        if valid_list[len(valid_list)-1]==data_max-1 or valid_list[0]==1:
            print('Train data from {:d} to {:d}'.format(train_list[0],train_list[valid_list[0]-2]),file=text_file)
            print('Train data from {:d} to {:d}'.format(train_list[0],train_list[valid_list[0]-2]))
        else:
            print('Train data from {:d} to {:d} and {:d} to {:d}'.format(train_list[0],train_list[valid_list[0]-2],train_list[valid_list[0]-1],train_list[len(train_list)-1]),file=text_file)
            print('Train data from {:d} to {:d} and {:d} to {:d}'.format(train_list[0],train_list[valid_list[0]-2],train_list[valid_list[0]-1],train_list[len(train_list)-1]))
        print('Valid data from {:d} to {:d}'.format(valid_list[0],valid_list[len(valid_list)-1]),file=text_file)
        print('Valid data from {:d} to {:d}'.format(valid_list[0],valid_list[len(valid_list)-1]))

        u = np.zeros((17,1))
        sigma = np.zeros((17,1))
        train_x = np.zeros((len(train_list),17))
        train_t = np.zeros((len(train_list),1))
        for i in range(0,17):
            for n in range(1,len(train_list)):
                train_x[n-1][i] = float(input_x[train_list[n-1]][i+1])
                u[i][0] = u[i][0] + float(input_x[train_list[n-1]][i+1])
            u[i][0] = u[i][0] / len(train_list)                   # Gaussian maximum likelihood mean
            for n in range(1,len(train_list)):
                sigma[i][0] = sigma[i][0] + pow(float(input_x[train_list[n-1]][i+1])-u[i][0],2)
            sigma[i][0] = sigma[i][0] / len(train_list)           # Gaussian maximum likelihood variance

        for n in range(1,len(train_list)):
            train_t[n-1][0] = float(input_t[n][1])

        phi = np.zeros((len(train_list),17))
        for i in range(0,17):
            for n in range(1,len(train_list)):
                phi[n-1][i] = exp((-1)*pow((train_x[n-1][i]-u[i][0]),2)/(2*pow(sigma[i][0],1)))

        A = (phi.T).dot(phi)
        B = np.linalg.inv(A).dot(phi.T)
        W_ML = B.dot(train_t)                               # Gaussian maximum likelihood W

        beta = 0
        for i in range(1,len(train_list)):
            beta = beta + pow((phi[i-1]).dot(W_ML)-train_t[i-1][0],2)   # beta comes from maximum likelihood
        beta = 1 / (beta/len(train_list))
        alpha = 1
        _lambda = alpha/beta

        B_MAP = np.linalg.inv(A+_lambda*np.identity(17)).dot(phi.T)
        W_MAP = B_MAP.dot(train_t)

        Erms = 0
        for n in range(0,len(valid_list)):
            x = np.zeros((1,17))
            for i in range(1,18):
                x[0][i-1] =exp((-1)*pow((float(input_x[valid_list[n]][i])-u[i-1][0]),2)/(2*pow(sigma[i-1][0],1)))

            y = x.dot(W_MAP) - float(input_t[valid_list[n]][1])
            # print(x)
            Erms = Erms + pow(y,2)
        Erms = pow(Erms/len(valid_list),0.5)
        print('Gaussian model with MAP (λ={:0.3f}), Erms = {:0.3f}\n'.format(_lambda,Erms),file=text_file)
        print('Gaussian model with MAP (λ={:0.3f}), Erms = {:0.3f}\n'.format(_lambda,Erms))
text_file.close