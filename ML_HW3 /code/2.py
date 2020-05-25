import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import csv
import os.path as osp
import math
from sklearn.svm import SVC

def read_file():
    input_x = []
    input_t = []
    with open(osp.join('..','x_train.csv'),newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            row_temp = []
            for i in range(len(row)):
                row_temp.append(float(row[i]))
            input_x.append(row_temp)
        # print(input_x[0])           #row 1
        csvfile.close
    with open(osp.join('..','t_train.csv'),newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            input_t.append(int(float(row[0])))     
        # print(input_t)          #row 1
        csvfile.close
    
    return input_x,input_t

def get_kernel(Xi,Xj,kernel_type):
    kernel_value = np.dot(Xi,Xj.T)
    if kernel_type=='linear':
        return kernel_value
    else:
        return kernel_value**2

def multiplier(input_x,input_t,kernel_type):
    if kernel_type=='linear':
        svm = SVC(kernel='linear',gamma='auto')
    else:
        svm = SVC(kernel='poly',degree=2,gamma='auto')
    svm.fit(input_x,input_t)                # input the train data
    alphas = np.abs(svm.dual_coef_)         # get the Lagrange multipliers(SV_num*2):coefficient for all 1-vs-1 classifiers.
    target_value = np.sign(svm.dual_coef_)  # get tn(SV_num*2)
    support_index = svm.support_            # get the support vectors index in train data(SV_num)
    support_vector = svm.support_vectors_   # get the support vectors directly(SV_num)
    # print(support_index)
    # print(len(input_x[support_index[0]]))
    # print(len(support_vector[0]))
    # print(alphas)
    # print(len(alphas))
    # print(len(alphas[1]))
    # print(target_value)
    return alphas,target_value,support_index,support_vector

# input whole_data (np.array(dxN)) output dimension-reduced data y (np.array(kxN))
def pca(whole_data, k):
    N = whole_data.shape[1] #num of date
    d = whole_data.shape[0] #num of parameter
    mean = np.zeros((N,1))
    scatter = np.zeros((d,d))

    mean = np.sum( whole_data, axis=1 )/N 

    for i in range(N):
        c = np.dot( (whole_data[:,i].reshape(d,1)-mean),(whole_data[:,i].reshape(d,1)-mean).T )
        scatter += c    


    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter)
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    transform_w = np.zeros((d,k))
    for i in range(k):
        transform_w[:,i] = eig_pairs[i][1]

    y = transform_w.T.dot(whole_data)
    return y
def get_b(pca_data,kernel_class,kernel_type):
    b = [0,0,0]
    for class_index in range(3):
        for i in range(len(kernel_class[class_index])):
            temp = 0
            for j in range(len(kernel_class[class_index])):
                temp += kernel_class[class_index][j][2]*kernel_class[class_index][j][1]*get_kernel(pca_data[int(kernel_class[class_index][i][0])],pca_data[int(kernel_class[class_index][j][0])],kernel_type)
            b[class_index] += (kernel_class[class_index][i][1] - temp)
        b[class_index] = b[class_index]/len(kernel_class[class_index])
    return b


# input_x:(Nxd) pca_data:(Nx2) test_x:(1x2)
def class_y(test_x,pca_data,input_t,b,kernel_class,kernel_type):
    y = [0,0,0]   # y0:0vs1 y1:0vs2 y2:1vs2


    for class_index in range(3):
        for i in range(len(kernel_class[class_index])):
            y[class_index] += kernel_class[class_index][i][2]*kernel_class[class_index][i][1]*get_kernel(test_x,pca_data[int(kernel_class[class_index][i][0])],kernel_type)
        y[class_index] += b[class_index]

    # for i in range(len(support_index)):

        # if input_t[support_index[i]] == 0:
        #     y[1] += alphas[0][i]*target_value[0][i]*get_kernel(test_x,pca_data[support_index[i]],kernel_type)
        #     y[2] += alphas[1][i]*target_value[1][i]*get_kernel(test_x,pca_data[support_index[i]],kernel_type)
        # elif input_t[support_index[i]] == 1:
        #     y[0] += alphas[0][i]*target_value[0][i]*get_kernel(test_x,pca_data[support_index[i]],kernel_type)
        #     y[2] += alphas[1][i]*target_value[1][i]*get_kernel(test_x,pca_data[support_index[i]],kernel_type)
        # else:
        #     y[0] += alphas[0][i]*target_value[0][i]*get_kernel(test_x,pca_data[support_index[i]],kernel_type)
        #     y[1] += alphas[1][i]*target_value[1][i]*get_kernel(test_x,pca_data[support_index[i]],kernel_type)

    p_y = np.sign(y)
    predict = 3
    if p_y[0]==1 and p_y[1]==1:
        predict = 0
    elif p_y[0]==-1 and p_y[2]==1:
        predict = 1
    elif p_y[1]==-1 and p_y[2]==-1:
        predict = 2
    # print(b)
    return predict

def plot_svc(draw_x1,draw_x2,draw_t,fig_name,sca_0,sca_1,sca_2,sca_sv):
    
    plt.figure(fig_name,figsize=(10,5))
    plt.suptitle(fig_name,fontsize=18)
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    colors = ('pink', 'lightblue', 'lightgreen','white')
    cmap = ListedColormap(colors[:len(np.unique(draw_t))])
    plt.pcolormesh(draw_x1,draw_x2,draw_t,cmap=cmap,alpha=1)
    sca0 = plt.scatter(sca_0[0],sca_0[1],s=15,c='red',marker='x',edgecolor='red',label='class 0')
    sca1 = plt.scatter(sca_1[0],sca_1[1],s=15,c='blue',marker='x',edgecolor='blue',label='class 1')
    sca2 = plt.scatter(sca_2[0],sca_2[1],s=15,c='green',marker='x',edgecolor='green',label='class 2')
    scasv = plt.scatter(sca_sv[0],sca_sv[1],s=20,facecolor='none',marker='o',edgecolor='black',label='Support Vector')
    plt.legend(handles = [sca0,sca1,sca2,scasv], loc='upper right')
    # line, = plt.plot(draw_x, draw_mean, color='red', linewidth = 2, label = 'Mean')
    # area = plt.fill_between(draw_x, draw_lowwer,draw_upper,facecolor='pink', interpolate=True, label = 'Variance')

    save_fig_path = osp.join('..','output','2',fig_name)
    plt.savefig(save_fig_path)
    plt.show()

def plot_sca(fig_name,sca_0,sca_1,sca_2,sca_sv):
    plt.figure(fig_name,figsize=(10,5))
    plt.suptitle(fig_name,fontsize=18)
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    sca0 = plt.scatter(sca_0[0],sca_0[1],s=15,c='red',marker='x',edgecolor='red',label='class 0')
    sca1 = plt.scatter(sca_1[0],sca_1[1],s=15,c='blue',marker='x',edgecolor='blue',label='class 1')
    sca2 = plt.scatter(sca_2[0],sca_2[1],s=15,c='green',marker='x',edgecolor='green',label='class 2')
    scasv = plt.scatter(sca_sv[0],sca_sv[1],s=20,facecolor='none',marker='o',edgecolor='black',label='Support Vector')
    plt.legend(handles = [sca0,sca1,sca2,scasv], loc='upper right')
    save_fig_path = osp.join('..','output','2',fig_name)
    plt.savefig(save_fig_path)
    plt.show()


if __name__ == '__main__':
    input_x,input_t = read_file()       # x:300*784 , t:300*1

    # use PCA and normalization
    pca_data = pca((np.array(input_x)).T,2).T
    mean_pca_data = np.mean(pca_data,axis=0)
    std_pca_data = np.std(pca_data,axis=0)
    normal_pca = np.zeros((len(pca_data),2))
    for i in range(normal_pca.shape[0]):
        normal_pca[i][0] = (pca_data[i][0]-mean_pca_data[0])/std_pca_data[0]
        normal_pca[i][1] = (pca_data[i][1]-mean_pca_data[1])/std_pca_data[1]
    # print(normal_pca)
    max_x1 = np.max(normal_pca[:,0])
    max_x2 = np.max(normal_pca[:,1])
    min_x1 = np.min(normal_pca[:,0])
    min_x2 = np.min(normal_pca[:,1])
    # print(max_x1,min_x1,max_x2,min_x2)

    # used to plot the fig
    mesh_size = 200
    draw_x1 = np.linspace(math.floor(min_x1),math.ceil(max_x1),mesh_size)
    draw_x2 = np.linspace(math.floor(min_x2),math.ceil(max_x2),mesh_size)
    draw_t = np.zeros((mesh_size,mesh_size))

    # classify 3 sca
    sca_0 = [[],[]]
    sca_1 = [[],[]]
    sca_2 = [[],[]]
    for i in range(normal_pca.shape[0]):
        if input_t[i] == 0:
            sca_0[0].append(normal_pca[i][0])
            sca_0[1].append(normal_pca[i][1])
        elif input_t[i] == 1:
            sca_1[0].append(normal_pca[i][0])
            sca_1[1].append(normal_pca[i][1])
        else:
            sca_2[0].append(normal_pca[i][0])
            sca_2[1].append(normal_pca[i][1])
    
    
    # get linear muliplier
    li_alphas,li_target_value,li_support_index,li_support_vector = multiplier(normal_pca,input_t,'linear')
    sca_sv =[[],[]]
    for i in range(len(li_support_index)):
        j = li_support_index[i]
        sca_sv[0].append(normal_pca[j][0])
        sca_sv[1].append(normal_pca[j][1])
    # plot_sca('Data with PCA(linear kernel)',sca_0,sca_1,sca_2,sca_sv)
    
    
    li_class = [[],[],[]]   # class 0vs1 0vs2 1vs2
    for i in range(len(li_support_index)):
        index = li_support_index[i]
        if int(input_t[index])==0:
            class_type = [0,1]
        elif int(input_t[index])==1:
            class_type = [0,2]
        else:
            class_type = [1,2]
        for j in range(2):
            if li_alphas[j][i]!=0:
                li_class[class_type[j]].append([li_support_index[i],li_target_value[j][i],li_alphas[j][i]])

    li_b = get_b(normal_pca,li_class,'linear')
    for i in range(mesh_size):
        for j in range(mesh_size):
            test = [draw_x1[i],draw_x2[j]]
            draw_t[j][i] = class_y(np.array(test),normal_pca,input_t,li_b,li_class,'linear')
    plot_svc(draw_x1,draw_x2,draw_t,'linear kernel',sca_0,sca_1,sca_2,sca_sv)
    
    # get polynomial(degree=2) muliplier
    po_alphas,po_target_value,po_support_index,po_support_vector = multiplier(normal_pca,input_t,'poly')   # 74
    sca_sv =[[],[]]
    for i in range(len(po_support_index)):
        j = po_support_index[i]
        sca_sv[0].append(normal_pca[j][0])
        sca_sv[1].append(normal_pca[j][1])
    # plot_sca('Data with PCA(polynomial kernel)',sca_0,sca_1,sca_2,sca_sv)

    po_class = [[],[],[]]   # class 0vs1 0vs2 1vs2
    for i in range(len(po_support_index)):
        index = po_support_index[i]
        if int(input_t[index])==0:
            class_type = [0,1]
        elif int(input_t[index])==1:
            class_type = [0,2]
        else:
            class_type = [1,2]
        for j in range(2):
            if po_alphas[j][i]!=0:
                po_class[class_type[j]].append([po_support_index[i],po_target_value[j][i],po_alphas[j][i]])
    # for i in range(len(po_class[0])):
    #     print('Class:{:f} Tn:{:f}'.format(input_t[int(po_class[0][i][0])],po_class[0][i][1]))
    # for i in range(len(po_class[1])):
    #     print('Class:{:f} Tn:{:f}'.format(input_t[int(po_class[1][i][0])],po_class[1][i][1]))
    # for i in range(len(po_class[2])):
    #     print('Class:{:f} Tn:{:f}'.format(input_t[int(po_class[2][i][0])],po_class[2][i][1]))
    # print(po_class)
    po_b = get_b(normal_pca,po_class,'poly')
    for i in range(mesh_size):
        for j in range(mesh_size):
            test = [draw_x1[i],draw_x2[j]]
            draw_t[j][i] = class_y(np.array(test),normal_pca,input_t,po_b,po_class,'poly')

    plot_svc(draw_x1,draw_x2,draw_t,'polynomial kernel',sca_0,sca_1,sca_2,sca_sv)
    
    
