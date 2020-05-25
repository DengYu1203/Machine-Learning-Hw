import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path as osp
import scipy.io
import math
k_mean_num = 3
train_data_size = 10
kmeans_center = []

def euclid_dis(x,y):
    sum = 0
    for i in range(1,len(x)):
        sum += (y[i]-x[i])**2
    return int(sum**0.5)

def cluster(x, kx):
    team = []
    for i in range(k_mean_num):
        team.append([])
    mid_dis = 99999999
    for i in range(train_data_size):
        for j in range(k_mean_num):
            distant = euclid_dis(x[i], kx[j])
            if distant < mid_dis:
                mid_dis = distant
                flag = j
        team[flag].append(x[i])
        mid_dis = 99999999
    return team

def re_seed(team, y):
    new_seed = []

    for i in range(0,k_mean_num):           # team num
        sum_x = np.linspace(0,0,8)
        target_class = [0,0,0]
        for j in range(0,len(team[i])):     # 1 data in one team
            target_class[team[i][j][0]] += 1
            for k in range(0,8):            # data's 8 attribute
                sum_x[k] += team[i][j][k]
        for k in range(0,8):
            sum_x[k] = (sum_x[k]/len(team[i]))
        sum_x[0] = target_class.index(max(target_class))
        new_seed.append(sum_x)
    return new_seed

def kmeans(x, kx, fig):
    team = cluster(x, kx)
    nkx = re_seed(team, kx)
    print('The {:d} in kmeans'.format(fig))
    for i in range(0,len(nkx)):
        print(nkx[i])
    
    # 判斷群集中心是否不再更動
    for i in range(0,len(nkx)):
        for j in range(1,len(nkx[i])):
            if math.fabs(nkx[i][j]-kx[i][j]) > 10**(-4):
                fig += 1
                kmeans(x,nkx,fig)
                return
    kmeans_center.clear()
    for i in range(0,len(nkx)):
        kmeans_center.append(nkx[i])


input_x = []    #
inputfile = osp.join('..','dataset','Pokemon.csv')
with open(inputfile,newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if row[1]=='Name':
            continue
        data = []
        data.append(row[1])
        if row[2] == 'Water':
            data.append(0)
        elif row[2] == 'Normal':
            data.append(1)
        elif row[2] == 'Psychic':
            data.append(2)
        for i in range(4,11):
            data.append(int(row[i]))
        if row[11] == 'TRUE':
            data.append(True)
        else:
            data.append(False)
        input_x.append(data)
    csvfile.close
# print(input_x[0])           #row 1
train_x = []
mu_var = np.zeros((2,8))    # row 1: mean, row 2 : sigma
for i in range(0,train_data_size):
    row = input_x[i]
    data = []
    for j in range(1,9):
        data.append(row[j])
        if j != 8:
            mu_var[0][j] += row[j+1]
    # if row[9] == True:
    #     data.append(1)
    #     mu_var[0][8] += 1
    # else:
    #     data.append(0)
    train_x.append(data)

for j in range(0,8):                    #mean
    mu_var[0][j] = mu_var[0][j] / train_data_size
# print(mu_var[0][8])
for i in range(0,train_data_size):
    for j in range(0,8):
        mu_var[1][j] += (train_x[i][j]-mu_var[0][j])**2

for j in range(0,8):                    #sigma
    mu_var[1][j] = (mu_var[1][j] / train_data_size)**0.5
# print(mu_var[1][8])
for i in range(0,train_data_size):
    for j in range(1,8):
        train_x[i][j] = (train_x[i][j] - mu_var[0][j])/mu_var[1][j]



kx = []     #initial guess k points
for i in range(k_mean_num):
    kx.append(train_x[ i * int(train_data_size/k_mean_num)])

kmeans(train_x,kx,fig=0)

print('Final kmeans centers:')
for i in range(0,len(kmeans_center)):
    print((kmeans_center[i]))


