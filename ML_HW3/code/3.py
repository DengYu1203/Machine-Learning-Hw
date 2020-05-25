import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import os.path as osp
import scipy.io
import math
import json
k_mean_num = 2
train_data_size = 10
kmeans_center = []
kmeans_team = []
gmm_iteration = 100
def read_file():
    path = osp.join('..','hw3_3.jpeg')
    img = cv2.imread(path)  # BGR
    data = []
    for i in range(img.shape[0]):       # img row:344
        for j in range(img.shape[1]):   # img col:500
            data.append(img[i][j])
            # print(img[i][j])
    # cv2.imshow('Hw3',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return data

def euclid_dis(x,y):
    sum = 0
    for i in range(len(x)):
        sum += (float(y[i])-float(x[i]))**2
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
        team[flag].append([x[i],i]) # team[k] = [[data],index]
        mid_dis = 99999999
    return team

def re_seed(team, y):
    new_seed = []

    for i in range(0,k_mean_num):           # team num
        sum_x = np.linspace(0,0,3)
        for j in range(0,len(team[i])):     # data in one team
            for k in range(0,3):            # data's 3 attribute(BGR)
                sum_x[k] += team[i][j][0][k]
        for k in range(0,3):
            # print(sum_x[k],len(team[i]))
            sum_x[k] = float(sum_x[k]/len(team[i]))
        new_seed.append(sum_x)
    return new_seed

def kmeans(x, kx, fig):
    team = cluster(x, kx)
    nkx = re_seed(team, kx)
    print('The {:d}-th time in kmeans(K={})'.format(fig,k_mean_num))
    for i in range(0,len(nkx)):
        print(nkx[i])
    
    # 判斷群集中心是否不再更動
    for i in range(0,len(nkx)):
        for j in range(len(nkx[i])):
            if math.fabs(nkx[i][j]-kx[i][j]) > 10**(-4):
                fig += 1
                kmeans(x,nkx,fig)
                return
    kmeans_center.clear()
    kmeans_team.clear()
    for i in range(0,len(nkx)):
        kmeans_center.append(nkx[i])
    for i in range(len(team)):
        kmeans_team.append(team[i])

def save_img(gmm_flag=False):
    cluster_img = np.zeros((344, 500, 3), dtype = "uint8")
    for i in range(len(kmeans_team)):
        pixel_color = [int(kmeans_center[i][0]),int(kmeans_center[i][1]),int(kmeans_center[i][2])]
        for j in range(len(kmeans_team[i])):
            img_index_i = int(kmeans_team[i][j][1]/500)
            img_index_j = int(kmeans_team[i][j][1]%500)
            for color in range(3):
                cluster_img[img_index_i][img_index_j][color] = pixel_color[color]
    if gmm_flag == False:
        fig_name = osp.join('..','output','3','K-means with K = '+str(k_mean_num)+'.png')
    else:
        fig_name = osp.join('..','output','3','GMM with K = '+str(k_mean_num)+'.png')
    cv2.imwrite(fig_name,cluster_img)

def gaussian(x,mean,cov):   # x and mean are row vectors
    # print(cov)
    cov_det = np.linalg.det(cov)
    # print(cov_det)
    a = 1/((((2*math.pi)**3)*cov_det)**0.5)
    # print(a)
    A = (np.array(x).reshape((3,1))-np.array(mean).reshape((3,1)))
    # print(A)
    B = np.dot(A.T,np.linalg.pinv(cov))
    b = -0.5*(np.dot(B,A))
    # print(b)

    return a*math.exp(b)

def plot_likelihood(mean_list,pi_list,cov_list,fig_name):
    b = np.linspace(0,255,100)
    g = np.linspace(0,255,100)
    r = np.linspace(0,255,100)
    # prob = np.zeros((100,100,100))
    gray = np.linspace(0,0,1000000)
    prob_gray = np.linspace(0,0,1000000)
    print('Ploting likelihood:')
    for i in range(100):
        for j in range(100):
            for k in range(100):
                sum_temp = 0
                for cluster_index in range(k_mean_num):
                    xn = np.array([b[i],g[j],r[k]])
                    sum_temp += pi_list[cluster_index]*gaussian(xn,mean_list[cluster_index],cov_list[cluster_index])
                # prob[i][j][k] = math.log1p(sum_temp)
                prob_gray[10000*i+100*j+k] = math.log1p(sum_temp)
                gray[10000*i+100*j+k] = 0.2989 * r[k] + 0.5870 * g[j] + 0.1140 * b[i]
    plt.figure(fig_name,figsize=(10,5))
    plt.suptitle(fig_name,fontsize=18)
    plt.xlabel('gray color',fontsize=14)
    plt.ylabel('ln(P)',fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # ax = Axes3D(fig)
    # ax.plot_surface(b, g, r,prob)
    # plt.show()
    line, = plt.plot(gray, prob_gray, color='red', linewidth = 2, label = 'log likelihood')
    plt.legend(handles = [line], loc='upper right')
    save_fig_path = osp.join('..','output','3',fig_name)
    plt.savefig(save_fig_path)



def GMM(img,first=True):
    mean_list = []
    pi_list = []
    cov_list = []
    # if first==True:
    for i in range(len(kmeans_team)):
        center = [float(kmeans_center[i][0]),float(kmeans_center[i][1]),float(kmeans_center[i][2])]
        center = np.array(center)
        mean_list.append(center)
        pi_list.append(len(kmeans_team[i])/train_data_size)
        temp = np.zeros((3,3))
        for j in range(len(kmeans_team[i])):
            xn = np.array(img[kmeans_team[i][j][1]])
            col_vec = xn.reshape((3,1)) - mean_list[i].reshape((3,1))
            # print(col_vec)
            temp += col_vec * col_vec.T
        cov_list.append(temp/len(kmeans_team[i]))
    # print(mean_list)
    # print(pi_list)
    # print(cov_list)
    # GMM(img,first=False)
    # else:
    total = 0
    r_temp = []
    
    for iter_index in range(gmm_iteration):
        print('The {}-th iteration in GMM(k={})'.format(iter_index+1,k_mean_num))
        # E-step
        r = np.zeros((train_data_size,k_mean_num))
        total = 0
        team = []
        for i in range(k_mean_num):
            team.append([])
        for i in range(train_data_size):
            r_temp.clear()
            total = 0
            for j in range(k_mean_num):
                temp = pi_list[j]*gaussian(img[i],mean_list[j],cov_list[j])
                r_temp.append(temp)
                total += temp
            for k in range(k_mean_num):
                r_temp[k] /= total
                r[i][k] = r_temp[k]
                
            cluster_max = max(r_temp)
            cluster_index = r_temp.index(cluster_max)
            # r.append([cluster_index,cluster_max,r_temp])   # r[data_num] = [k_cluster_index,r(z)]
            team[cluster_index].append([img[i],i])
        kmeans_team.clear()
        for i in range(len(team)):
            kmeans_team.append(team[i])

        # M-step
        kmeans_center.clear()
        for i in range(k_mean_num):
            Nk = 0
            mean_sum = np.zeros((1,3))
            cov_sum = np.zeros((3,3))
            for j in range(train_data_size):
                r_znk = r[j][i]
                Nk += r_znk
                mean_sum += r_znk*np.array(img[j])
            mean_list[i] = mean_sum/Nk
            # print(mean_list[i].tolist())
            kmeans_center.append(mean_list[i].tolist()[0])
            pi_list[i] = Nk/train_data_size
            for j in range(train_data_size):
                r_znk = r[j][i]
                cov_sum += r_znk*(np.array(img[j]).reshape((3,1))-mean_list[i].reshape((3,1)))*(np.array(img[j]).reshape((3,1))-mean_list[i].reshape((3,1))).T
            cov_list[i] = cov_sum / Nk
        print('mean:\n{}'.format(mean_list))
        print('pi:\n{}'.format(pi_list))
        print('covariance:\n{}'.format(cov_list))
    save_img(gmm_flag=True)
    fig_name = 'GMM log likelihood (k='+str(k_mean_num)+')'
    plot_likelihood(mean_list,pi_list,cov_list,fig_name)
    
    path = osp.join('..','output','3','GMM(k='+str(k_mean_num)+')'+'.txt')
    with open(path,'w') as file:
        # root = []
        # data = {}
        # data['mean'] = []
        # data['mean'].append(kmeans_center)
        # data['pi'] = []
        # data['pi'].append(pi_list)
        # data['covariance'] = []
        # data['covariance'].append(cov_list.tolist())
        # data['team'] = []
        # data['team'].append(kmeans_team)
        # root.append(data)
        # json.dump(root,file,indent=1)
        file.write('mean\n')
        print(kmeans_center,file=file)
        file.write('Pi\n')
        print(pi_list,file=file)
        file.write('covariance\n')
        print(cov_list,file=file)
        # file.write('team\n')
        # print(kmeans_team,file=file)
        file.close()
        
    return



if __name__ == '__main__':
    img = read_file()   # 344*500 = 17200 pixels
    train_data_size = len(img)
    k_mean_list = [3,5,7,10]
    # k_mean_list = [2]
    for k_mean_num in k_mean_list:
        kx = []     # initial guess k points
        for i in range(k_mean_num):
            kx.append(img[ i * int(train_data_size/k_mean_num)])
        # print(kx)
        kmeans(img,kx,fig=0)
        print('\nFinal kmeans centers:')
        print('------------------------')
        for i in range(len(kmeans_center)):
            print('K={} cluster:{}'.format(k_mean_num,i))
            print('Center BGR {}'.format(kmeans_center[i]))
            print('{} datas in cluser {}'.format(len(kmeans_team[i]),i))
            # print('team {}, data-index {}, data {}'.format(kmeans_team[i][0][0],kmeans_team[i][0][1],img[kmeans_team[i][0][1]]))
            print('------------------------')
        # save_img()
        GMM(img)
