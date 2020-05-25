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
gmm_iteration = 15
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
        # r = []
        r = np.zeros((train_data_size,k_mean_num))
        # r.clear()
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
                # print(r_temp)
            # r.append(r_temp)   # r[data_num] = [k_cluster_index,r(z)]
            cluster_max = max(r_temp)
            cluster_index = r_temp.index(cluster_max)
            # r.append([cluster_index,cluster_max,r_temp])   # r[data_num] = [k_cluster_index,r(z)]
            
            # print(r[i][0])
            team[cluster_index].append([img[i],i])
        # print(len(r))
        kmeans_team.clear()
        for i in range(len(team)):
            kmeans_team.append(team[i])
        # for i in range(train_data_size):
        #     print(r[i])
        # M-step
        kmeans_center.clear()
        for i in range(k_mean_num):
            Nk = 0
            mean_sum = np.zeros((1,3))
            cov_sum = np.zeros((3,3))
            # for j in range(len(team[i])):
                # data_index = int(team[i][j][1])
            for j in range(train_data_size):
                # data_index = j
                # print(j)
                r_znk = r[j][i]
                # print(j,i,r[j][i])
                Nk += r_znk
                mean_sum += r_znk*np.array(img[j])
            mean_list[i] = mean_sum/Nk
            # print(mean_list[i].tolist())
            kmeans_center.append(mean_list[i].tolist()[0])
            pi_list[i] = Nk/train_data_size
            # for j in range(len(team[i])):
                # data_index = int(team[i][j][1])
            for j in range(train_data_size):
                # data_index = j
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
    # k_mean_list = [3,5,7,10]
    k_mean_list = [10,7,5,3]
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
    # k_mean_num = 3
    # fig_name = 'GMM log likelihood (k='+str(k_mean_num)+')'
    # mean_list = [[162.15714247909744, 144.19622708700263, 125.29357954409109], [13.75648185288342, 110.17955403653397, 136.96936193607362], [248.50813046696018, 165.47860602496604, 70.74271236838527]]
    # pi_list = [0.3569271833872814, 0.4375760919603721, 0.1880045230262889]
    # cov_list = [[[2554.46671877, 1547.41435895,  661.47047903],
    #    [1547.41435895, 2313.83337247, 3070.84931997],
    #    [ 661.47047903, 3070.84931997, 5672.42302612]],
    #    [[ 209.22792309,  260.14068878,  271.98241927],
    #    [ 260.14068878, 2903.81147499, 3485.38870368],
    #    [ 271.98241927, 3485.38870368, 4663.35017472]],
    #    [[ 38.6913093 ,  93.1980221 ,  86.47415764],
    #    [ 93.1980221 , 444.8707915 , 449.746818  ],
    #    [ 86.47415764, 449.746818  , 634.55632719]]]

    # k_mean_num = 5
    # fig_name = 'GMM log likelihood (k='+str(k_mean_num)+')'
    # mean_list = [[183.56331171591182, 188.4745368577859, 199.28791047550126], [142.91162227063424, 112.46176235916563, 73.92882638419749], [10.033435084026243, 52.07799716076719, 50.27958529145946], [13.569758949204388, 135.71668543080816, 175.51466708126532], [248.9503908555821, 167.61533107972846, 74.71237828283316]]
    # pi_list = [0.13610579129440933, 0.2205204360597012, 0.12237487894367272, 0.28919316488593727, 0.18905001488532083]
    # cov_list = [[[1303.6877326 , 1100.4896578 ,  856.36569206],
    #    [1100.4896578 , 1059.07513872,  913.53932425],
    #    [ 856.36569206,  913.53932425,  919.48646813]],
    #    [[ 3280.98539757,   749.20910733, -1511.69698964],
    #    [  749.20910733,   926.56092999,   843.08256735],
    #    [-1511.69698964,   843.08256735,  2922.31178637]],
    #    [[ 152.34960192,  271.23531612,  264.05916957],
    #    [ 271.23531612, 1125.74023181, 1106.48961152],
    #    [ 264.05916957, 1106.48961152, 1165.84763853]],
    #    [[ 172.98555908,  174.38133322,  178.57268685],
    #    [ 174.38133322, 1722.33745134, 1547.50102971],
    #    [ 178.57268685, 1547.50102971, 1665.05760765]],
    #    [[ 33.37588159,  82.7287467 ,  71.4882142 ],
    #    [ 82.7287467 , 468.09104871, 508.80258793],
    #    [ 71.4882142 , 508.80258793, 764.34451783]]]

    # k_mean_num = 7
    # fig_name = 'GMM log likelihood (k='+str(k_mean_num)+')'
    # mean_list = [[202.44698382412614, 205.50589797720218, 214.29034467086933], [3.5460677205743063, 39.078263013600285, 37.105532199943006], [122.16455281971525, 134.0089947074888, 141.48630241410524], [16.32089260710106, 92.53522048571101, 111.59115428594865], [17.395842373266305, 153.79653698447896, 198.5041790132411], [248.58902502047047, 167.77733410720526, 76.15991209243978], [174.95991246266667, 107.55681734590347, 34.18508197116257]]
    # pi_list = [0.09023569561460372, 0.08926770523561733, 0.12327582275684165, 0.12534483679054623, 0.18868764301168564, 0.19162149388571328, 0.11921656549410065]
    # cov_list = [[[882.43903252, 700.01825378, 379.58321038],
    #    [700.01825378, 663.77842693, 481.3077125 ],
    #    [379.58321038, 481.3077125 , 549.35291179]],
    #    [[ 26.11860887,  31.48093865,  17.63735465],
    #    [ 31.48093865, 664.47558908, 629.63277154],
    #    [ 17.63735465, 629.63277154, 685.11700087]],
    #    [[1339.67090547,  472.34627564,  171.92871296],
    #    [ 472.34627564,  736.97111019,  841.1208908 ],
    #    [ 171.92871296,  841.1208908 , 1211.88202586]],
    #    [[ 258.97865393,   -7.251406  , -248.42989573],
    #    [  -7.251406  ,  788.46168346,  747.65596462],
    #    [-248.42989573,  747.65596462, 1140.66150726]],
    #    [[ 200.57224711,   43.98517486,   33.74404234],
    #    [  43.98517486, 1225.52769331,  813.0575562 ],
    #    [  33.74404234,  813.0575562 ,  720.11801214]],
    #    [[ 38.91636814,  79.17920298,  53.93708038],
    #    [ 79.17920298, 473.83242603, 502.35546545],
    #    [ 53.93708038, 502.35546545, 732.2294968 ]],
    #    [[2151.2337924 ,  956.59021063, -522.84306982],
    #    [ 956.59021063,  615.34771425,    5.91668896],
    #    [-522.84306982,    5.91668896,  563.88303362]]]
    
    # k_mean_num = 10
    # fig_name = 'GMM log likelihood (k='+str(k_mean_num)+')'
    # mean_list = [[244.52566053404468, 207.17465765926966, 158.29345610923068], [248.79381753490986, 166.8277423969206, 76.86708289932085], [147.12807218428807, 98.48663024669439, 47.68183896990584], [191.65488890690978, 197.8026950443386, 211.40167927674833], [20.305749080769647, 76.4838232595368, 83.37477873496421], [22.95599376443516, 169.16612628170935, 213.08201330224404], [124.05574501222442, 134.17660014761202, 141.44175535821833], [0.18189437710171188, 33.74233203822452, 32.719117669879054], [226.55440496926812, 135.45807051350664, 22.111963577278306], [5.24113174469893, 118.17694253639014, 158.52874843900227]]
    # pi_list = [0.023486011244250024, 0.15330216424922102, 0.08509253247791053, 0.08648102675359083, 0.11853773935403691, 0.1089903315913327, 0.10616464831386963, 0.05048857467335881, 0.062135049971022795, 0.11994626245488273]
    # cov_list = [[[ 108.0746639 ,  172.45504754,  260.28277303],
    #    [ 172.45504754,  657.79292887, 1082.48150143],
    #    [ 260.28277303, 1082.48150143, 2025.41449274]],
    #    [[ 36.92094054,  95.6048123 ,  81.16272358],
    #    [ 95.6048123 , 436.39090712, 390.62019098],
    #    [ 81.16272358, 390.62019098, 387.70497486]],
    #    [[2456.32296832, 1034.28551525, -255.38901637],
    #    [1034.28551525,  644.33325047,  123.43857684],
    #    [-255.38901637,  123.43857684,  420.61096055]],
    #    [[709.66289792, 618.76964259, 476.51584435],
    #    [618.76964259, 621.34680175, 525.98133584],
    #    [476.51584435, 525.98133584, 525.13669279]],
    #    [[ 214.02955412,  295.93140727,  271.89113199],
    #    [ 295.93140727, 1194.38601262, 1281.83901024],
    #    [ 271.89113199, 1281.83901024, 1696.75489121]],
    #    [[215.06843679, -99.637964  , -91.93838699],
    #    [-99.637964  , 954.50340209, 568.66184215],
    #    [-91.93838699, 568.66184215, 492.6678191 ]],
    #    [[1215.38882261,  416.08113201,   91.73594924],
    #    [ 416.08113201,  718.43337592,  797.6772216 ],
    #    [  91.73594924,  797.6772216 , 1133.87006576]],
    #    [[2.93346077e-01, 1.60239764e-01, 8.16992793e-03],
    #    [1.60239764e-01, 6.65338594e+02, 6.56817394e+02],
    #    [8.16992793e-03, 6.56817394e+02, 6.96447665e+02]],
    #    [[423.4250572 , 487.95224579, 290.69244078],
    #    [487.95224579, 634.89253648, 396.41420812],
    #    [290.69244078, 396.41420812, 277.52867546]],
    #    [[ 31.11796515,  16.5155425 ,  22.07608955],
    #    [ 16.5155425 , 802.80050196, 533.35037702],
    #    [ 22.07608955, 533.35037702, 559.67009783]]]
    # plot_likelihood(mean_list,pi_list,cov_list,fig_name)
