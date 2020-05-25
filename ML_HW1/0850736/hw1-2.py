from math import pow
import matplotlib.pyplot as plt
import numpy as np
import csv

input_x = []    #1096 data
input_t = []    #1096 data
with open('../Dataset/dataset_X.csv',newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        input_x.append(row)
    # print(input_x[1])           #row 1
    csvfile.close
with open('../Dataset/dataset_T.csv',newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        input_t.append(row)     
    # print(input_t[1])          #row 1
    csvfile.close

data_max = len(input_x)
train = int((data_max-1) * 6 /8)
valid = train + 1
# train = data_max - 1
# valid = 1

print('Train data from 1 to {:d}'.format(train))
print('Valid data from {:d} to {:d}'.format(valid,data_max-1))

train_t = np.zeros((train,1))
phi = np.zeros((train,171))
for n in range(1,train+1):
    phi[n-1][0] = 1
    index = 18
    for i in range(1,18):
        phi[n-1][i] = float(input_x[n][i])
        for j in range(i,18):
            phi[n-1][index] = float(input_x[n][i])*float(input_x[n][j])
            index = index + 1
    train_t[n-1][0] = float(input_t[n][1])

A = (phi.T).dot(phi)
B =  np.linalg.inv(A).dot(phi.T)
W = B.dot(train_t)

Erms = 0
for n in range(valid,data_max):
    x = np.zeros((1,171))
    x[0][0] = 1
    for i in range(1,18):
        x[0][i] = float(input_x[n][i])
    m = 18
    for k in range(1,18):
        for j in range(k,18):
            x[0][m] = float(input_x[n][k]) * float(input_x[n][j])
            m = m + 1
    y = x.dot(W) - float(input_t[n][1])
    # print(x)
    Erms = Erms + pow(y,2)
Erms = pow(Erms/(data_max-valid),0.5)
print('1 \n   (a) M = 2 , Erms = {:0.3f}'.format(Erms))
# print(W.T)
# print(B)

