import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os.path as osp


train_size = 5
faces_num = 5
class_sam = 10
target = []
target_test = []


test_raw = []
Test_result = []

def load_img(dir_path):
    global test_raw
    train = []
    test = []
    
    for i in range(0,faces_num):
        train_list = random.sample(range(0,10),train_size)
        for j in range(0,10):
            path = osp.join(dir_path,'s'+str(i+1),str(j+1)+'.pgm')
            img = cv2.imread(path,0)
            img_flatten = img.flatten() / 255.0
            if j in train_list:
                train.append(img_flatten)
            else:
                test.append(img_flatten)
                test_raw.append(img)

    return train, test


def softmax(x,w):
    a = np.dot(w,x.T)

    a_exp = np.exp(a)
    sum_exp = np.array( np.sum(a_exp,axis=0) )

    row, col = a.shape
    for i in range(row):
        for j in range(col):
            a[i][j] = a_exp[i][j] / sum_exp[j]
    
    return a


def accur(w, t, x, test=False):
    global Test_result
    correct = 0
    total_N = x.shape[0]
    a = softmax(x,w) 
    # print(a)
    result = np.where(a == np.amax(a, axis=0))

    if (test):
        Test_result = result[0]
    
    for index, c in enumerate(result[0]):
        if( t[index] == (c+1) ):
            correct+=1
    
    accuracy = correct / total_N
    return accuracy


def plot(error, accu, counter,method_type):
    save_path = osp.join('..','output','2',str(method_type)+'E(w)')
    fig = plt.figure('E(w)')
    fig.suptitle('E(w) curve',fontsize=25)
    plt.xlabel('iteration',fontsize = 16)
    plt.ylabel('E(w)',fontsize = 16)
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['figure.titlesize'] = 30
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.grid(True)
    plt.plot(range(counter),error, linestyle='-', zorder = 1, color = 'pink', linewidth=1)
    plt.plot(range(counter),error,'ro', ms=4)
    plt.savefig(save_path)

    save_path = osp.join('..','output','2',str(method_type)+'Accuracy')
    fig2 = plt.figure('Accuracy')
    fig2.suptitle('Accuracy curve',fontsize=25)
    plt.xlabel('iteration',fontsize = 16)
    plt.ylabel('Accu',fontsize = 16)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['figure.titlesize'] = 30
    plt.grid(True)
    plt.plot(range(counter),accu, linestyle='-', zorder = 1, color = 'blue', linewidth=1)
    plt.plot(range(counter),accu,'bo', ms=4)
    plt.savefig(save_path)
    plt.show()
    

iteration = 50
step = 0.001 
threshold = pow(10,-3)

def gradient_descent(img_train, target):

    total_N = np.array(img_train).shape[0]
    w = np.zeros((faces_num,len(img_train[0])))
    counter = 0
    error_list = []
    accu_list = []
    last_E = 0

    while (True):
        E = 0
        Gradient = np.zeros(w.shape)
        a = softmax(np.array(img_train),w)
        # print(np.max(img_train),np.max(w))

        for c in range(faces_num):
            grad_e = 0
            for n in range(total_N):

                if (math.isnan(a[c][n])):
                    print('NAN OCCURRED')
                    print('The learning rate is ',step)
                    print('The error is:\n',error_list)
                    print('The accuracy is\n',accu_list)
                    exit(-1)
            
                t = ( target[n] == (c+1) )
                error = t * np.log(a[c][n])
                grad_e += (a[c][n]-t)* img_train[n]
                
                E -= error
            Gradient[c] = grad_e
        error_list.append(E)
        w = w - step*Gradient

        counter += 1

        accuracy = accur(w,target,np.array(img_train))
        accu_list.append(accuracy)
        
        stop = ( math.fabs( ((error_list[-1]-last_E)/(last_E+0.0001))) < threshold )

        if ( stop  or counter >= iteration ):

            plot(error_list, accu_list, counter,'Gradient_')
            return w
        last_E = E



def result(target_test,accu_test,method_type):
    global test_raw
    img_cut =[test_raw[i:i+5] for i in range(0,len(test_raw),5)]
    target = [Test_result[i:i+5] for i in range(0,len(Test_result), 5)]

    h,w = test_raw[0].shape
    p = 50
    indent = 5
    H = 5*h + indent*4 + p*2 + 20
    W = 5*w + indent*4 + p*2
    img=np.zeros((H,W),np.uint8)
    img.fill(75)
 
    # print(img_cut[-1])
    accu = 'Accuracy: '+str(accu_test)

    cv2.putText(img,accu, (int(W/2)-60,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for index,row in enumerate(img_cut):
        for i in range(len(row)):
            img_t = img_cut[index][i]
            cv2.putText(img_t, 'c:%d'%(target[index][i]+1), (60,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 100), 1)
            img[20+p+h*index+indent*index:20+p+h*index+indent*index+h,(i*w+i*5)+50:((i+1)*w+i*5)+50] = img_t
    img_path = osp.join('..','output','2',str(method_type)+'face test.jpg')
    cv2.imwrite(img_path,img)
    cv2.imshow('face test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def newton(img_train, target, d):

    total_N = np.array(img_train).shape[0]
    w = np.zeros((faces_num,len(img_train[0])))

    counter = 0
    error_list = []
    accu_list = []
    w_list = [] 
    last_E = 0

    
    #calculate for gradient (class x data#)
    while (True):
        E = 0
        Gradient = np.zeros(w.shape)
        a = softmax(np.array(img_train),w)
        # print(np.max(img_train),np.max(w))

        for c in range(faces_num):
            grad_e = 0
            for n in range(total_N):

                if (math.isnan(a[c][n])):
                    print('NAN OCCURRED')
                    exit(-1)
            
                t = ( target[n] == (c+1) )
                error = t * np.log(a[c][n])
                grad_e += (a[c][n]-t)* img_train[n]
                
                E -= error
            Gradient[c] = grad_e
        error_list.append(E)
        w_list.append(w)
        
        I = np.identity(d)
        Hj = []

        for k in range(faces_num):
            Hessian = np.zeros((d,d))
            for n in range(total_N):
                out_product = np.dot( np.reshape( img_train[n],(len(img_train[n]),1) ) ,np.reshape( img_train[n],(len(img_train[n]),1) ).T)
                v = a[k][n]*(1-a[k][n])
                Hessian += (v * out_product)

            Hj.append(Hessian)
        for k in range(faces_num):
            w[k] = w[k] - np.dot( Gradient[k] ,np.linalg.pinv(Hj[k]) ) 

        counter += 1

        accuracy = accur(w,target,np.array(img_train))
        accu_list.append(accuracy)

        if ( counter >= iteration ):
            plot(error_list, accu_list, counter,'Newton_')
            return w
        last_E = E



def pca(whole_data, k,train_size):
    N = train_size #num of date
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



dir_path = osp.join('..','dataset','Faces')
img_train,img_test = load_img(dir_path)

# Gradient descent
for i in range(faces_num):
    for j in range(train_size):
        target.append(i+1)
        target_test.append(i+1)
    
w = gradient_descent(img_train, target)
accu_test = accur(w, target_test, np.array(img_test), True)
result(target_test,accu_test,'Gradient_')



# Newton Method
k_list = [2,5,10]
for k in k_list:
    whole_data = np.array(img_train + img_test).T
    N = whole_data.shape[1]
    y = pca(whole_data,k,25)

    y_train = y[:,:25]
    y_test = y[:,25:N]

    for i in range(faces_num):
        for j in range(train_size):
            target.append(i+1)
            target_test.append(i+1)

    #convet to list of array 
    y_train = (y_train.T).tolist()
    y_test = (y_test.T).tolist()
    for i in range(len(y_train)):
        y_train[i] = np.array(y_train[i])
        y_test[i] = np.array(y_test[i])   

    w = newton(y_train, target, k)
    accu_test = accur(w, target_test, np.array(y_test), True)
    result(target_test,accu_test,'Newton_k='+str(k)+'_')

