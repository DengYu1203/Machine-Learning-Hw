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
# print(len(input_x))
# print(len(input_t))

data_max = len(input_x)
# train = int((data_max-1) * 7 /8)
# valid = train + 1

train = data_max - 1
valid = 1

print('Train data from 1 to {:d}'.format(train))
print('Valid data from {:d} to {:d}'.format(valid,data_max-1))
A = np.zeros((18,18))   #(row,col)
B = np.zeros((18,1))
A[0][0] = 1


for i in range(1,18):
    sum = 0
    c = 0
    for j in range(1,train):
        sum = sum + float(input_x[j][i])
        c = c + float(input_t[j][1])
    A[0][i] = (sum/(train))
    A[i][0] = (sum/(train))
    B[0][0] = (c/(train))
    
    for k in range(1,18):
        sum = 0
        c = 0
        for j in range(1,train):
            sum = sum + float(input_x[j][i])*float(input_x[j][k])
            c = c + float(input_t[j][1])*float(input_x[j][i])
        A[i][k] = (sum/(train))
        B[i][0] = (c/(train))


# print(A)
# print(B)

W = np.linalg.inv(A).dot(B)
# print(W)
Erms = 0
for n in range(valid,data_max):
    x = np.zeros((1,18))
    x[0][0] = 1
    for i in range(1,18):
        x[0][i] = float(input_x[n][i])
    # print(x)
    y = x.dot(W) - float(input_t[n][1])
    # print(y)
    Erms = Erms + pow(y,2)
Erms = pow(Erms/(data_max-valid),0.5)
print('1 \n   (a) M = 1 , Erms = {:0.3f}'.format(Erms))
# print(W.T)

