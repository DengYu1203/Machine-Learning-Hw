import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path as osp
import scipy.io
import math
k_near_num = 3
train_size = 120

def euclid_dis(x,y):
    sum = 0
    for i in range(1,len(x)):
        sum += (y[i]-x[i])**2
    return int(sum**0.5)

def k_nearest(test_x, train_x,k):
    dis = []
    for i in range(0,len(train_x)):
        dis.append([euclid_dis(test_x,train_x[i]),i])
    dis.sort()

    data = []
    for i in range(0,k):
        data.append(dis[i])
    return data
    
# input whole_data (np.array(dxN)) output dimension-reduced data y (np.array(kxN)) 
def pca(whole_data, k,train_size):
    
    # print(whole_data.shape)
    # N = whole_data.shape[1] #num of date
    N = train_size
    d = whole_data.shape[0] #num of parameter
    mean = np.zeros((N,1))
    scatter = np.zeros((d,d))


    mean = np.sum( whole_data, axis=1 )/N 
 
    for i in range(N):
        c = np.dot( (whole_data[:,i].reshape(d,1)-mean),(whole_data[:,i].reshape(d,1)-mean).T )
        # print(c.shape)
        scatter += c    

    # print(scatter, scatter.shape)

    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)


    ## get first k eigenvetor to form transform matrix
    transform_w = np.zeros((d,k))
    for i in range(k):
        transform_w[:,i] = eig_pairs[i][1]

    y = (transform_w.T.dot(whole_data)).T
    train_pca = np.zeros((train_size,len(y[0])))
    test_pca = np.zeros((len(y)-train_size,len(y[0])))
    # print(len(y[0]))
    for i in range(0,train_size):
        for j in range(len(y[0])):
            train_pca[i][j] = y[i][j]
    for i in range(train_size,len(y)):
        for j in range(len(y[0])):
            test_pca[i-train_size][j] = y[i][j]
    # print(train_pca)
    # print(test_pca)
    return train_pca,test_pca

def plot_fig(test_x,train_x,fig_name):
    k_list = []
    accu_list = []
    print(fig_name)
    for k_near_num in range(1,11):
        accuracy = 0
        for test_index in range(0,len(test_x)):
            
            test = k_nearest(test_x[test_index],train_x,k_near_num)
            predict_type = [0,0,0]
            for i in range(0,len(test)):
                # print(train_x[test[i][1]])
                predict_type[int(train_x[test[i][1]][0])] += 1
            if test_x[test_index][0] == predict_type.index(max(predict_type)):
                accuracy += 1
        k_list.append(k_near_num)
        accu_list.append(accuracy/len(test_x))
        print('K = {:d} , Accuracy:{:f}'.format(k_near_num,accuracy/len(test_x)))
        
    # fig_name = 'K-nearest-neighbor (using all features)'
    plt.figure(fig_name,figsize=(10,5))
    plt.xlabel('K', fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 16)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)

    line1, = plt.plot(k_list, accu_list, color = '#1FC2DE', linewidth = 1)

    save_path = osp.join('..','output','3',fig_name)
    plt.savefig(save_path)
    plt.show()


input_x = []    # Type Total HP atk ... Legendary(1+9)

inputfile = osp.join('..','dataset','Pokemon.csv')
with open(inputfile,newline='') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if row[1]=='Name':
            continue
        data = []
        # data.append(row[1])
        if row[2] == 'Water':
            data.append(0)
        elif row[2] == 'Normal':
            data.append(1)
        elif row[2] == 'Psychic':
            data.append(2)
        for i in range(3,11):
            data.append(int(row[i]))

        if row[11] == 'True':
            data.append(1)
        else:
            data.append(0)
        input_x.append(data)
    csvfile.close

# for i in range(0,train_size):
#     print(input_x[i])

mean = np.linspace(0,0,len(input_x[0])-1)  # type 不列入計算
sigma = np.linspace(0,0,len(input_x[0])-1)  # type 不列入計算

train_x = []
test_x = []
# find mean and sigma
for i in range(0,train_size):
    for j in range(0,len(mean)):
        mean[j] += input_x[i][j+1]
for i in range(0,len(mean)):
    mean[i] = mean[i]/train_size
for i in range(0,train_size):
    for j in range(0,len(sigma)):
        sigma[j] += (input_x[i][j+1]-mean[j])**2
for i in range(0,len(sigma)):
    sigma[i] = (sigma[i]/train_size)**0.5
# print(mean[len(mean)-1])
# print(sigma[len(sigma)-1])

for i in range(0,train_size):
    data = []
    data.append(input_x[i][0])
    for j in range(0,len(mean)):
        data.append((input_x[i][j+1]-mean[j])/sigma[j])
    train_x.append(data)
# print(train_x)
for i in range(train_size,len(input_x)):
    data = []
    data.append(input_x[i][0])
    for j in range(0,len(mean)):
        data.append((input_x[i][j+1]-mean[j])/sigma[j])
    test_x.append(data)
# print(test_x)


fig_name = 'K-nearest-neighbor (using 9 features)'
# save_path = osp.join('..','output','3',fig_name)
plot_fig(test_x,train_x,fig_name)


# train_x_pca = np.zeros((len(train_x[0])-1,len(train_x)))
# test_x_pca = np.zeros((len(test_x[0])-1,len(test_x)))
# for i in range(len(train_x[0])-1):
#     for j in range(len(train_x)):
#         train_x_pca[i][j] = train_x[j][i+1]
# for i in range(len(test_x[0])-1):
#     for j in range(len(test_x)):
#         test_x_pca[i][j] = test_x[j][i+1]

train_pca = np.zeros((len(train_x[0])-1,len(input_x)))
for i in range(len(train_x[0])-1):
    for j in range(train_size):
        train_pca[i][j] = train_x[j][i+1]
for i in range(len(test_x[0])-1):
    for j in range(train_size,len(input_x)):
        train_pca[i][j] = test_x[j-train_size][i+1]

fig_name = 'K-nearest-neighbor (using 7 features)'
# save_path = osp.join('..','output','3',fig_name)
# train_pca_7 = (pca(np.array(train_x_pca),7,train_size)).T
# test_pca_7 = (pca(np.array(test_x_pca),7,train_size)).T
train_pca_7,test_pca_7 = pca(np.array(train_pca),7,train_size)

train_pca_7_in = np.zeros((len(train_pca_7),len(train_pca_7[0])+1))
test_pca_7_in = np.zeros((len(test_pca_7),len(test_pca_7[0])+1))
for i in range(len(train_pca_7_in)):
    train_pca_7_in[i][0] = train_x[i][0]
    for j in range(len(train_pca_7[0])):
        train_pca_7_in[i][j+1] = train_pca_7[i][j]
for i in range(len(test_pca_7_in)):
    test_pca_7_in[i][0] = test_x[i][0]
    for j in range(len(test_pca_7[0])):
        test_pca_7_in[i][j+1] = test_pca_7[i][j]

plot_fig(test_pca_7_in,train_pca_7_in,fig_name)

fig_name = 'K-nearest-neighbor (using 6 features)'
# train_pca_6 = (pca(np.array(train_x_pca),6)).T
# test_pca_6 = (pca(np.array(test_x_pca),6)).T
train_pca_6,test_pca_6 = pca(np.array(train_pca),6,train_size)
train_pca_6_in = np.zeros((len(train_pca_6),len(train_pca_6[0])+1))
test_pca_6_in = np.zeros((len(test_pca_6),len(test_pca_6[0])+1))
for i in range(len(train_pca_6_in)):
    train_pca_6_in[i][0] = train_x[i][0]
    for j in range(len(train_pca_6[0])):
        train_pca_6_in[i][j+1] = train_pca_6[i][j]
for i in range(len(test_pca_6_in)):
    test_pca_6_in[i][0] = test_x[i][0]
    for j in range(len(test_pca_6[0])):
        test_pca_6_in[i][j+1] = test_pca_6[i][j]

plot_fig(test_pca_6_in,train_pca_6_in,fig_name)

fig_name = 'K-nearest-neighbor (using 5 features)'
# train_pca_5 = (pca(np.array(train_x_pca),5)).T
# test_pca_5 = (pca(np.array(test_x_pca),5)).T
train_pca_5,test_pca_5 = pca(np.array(train_pca),5,train_size)
train_pca_5_in = np.zeros((len(train_pca_5),len(train_pca_5[0])+1))
test_pca_5_in = np.zeros((len(test_pca_5),len(test_pca_5[0])+1))
for i in range(len(train_pca_5_in)):
    train_pca_5_in[i][0] = train_x[i][0]
    for j in range(len(train_pca_5[0])):
        train_pca_5_in[i][j+1] = train_pca_5[i][j]
for i in range(len(test_pca_5_in)):
    test_pca_5_in[i][0] = test_x[i][0]
    for j in range(len(test_pca_5[0])):
        test_pca_5_in[i][j+1] = test_pca_5[i][j]

plot_fig(test_pca_5_in,train_pca_5_in,fig_name)

