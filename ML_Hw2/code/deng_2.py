import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os.path as osp
data_path = osp.join('..','dataset','Faces')

train_size = 5
class_num = 5
class_sam = 10
target = []
target_test = []
itera_num = 50
step_size = 0.001 
thres = 10**(-3)
# step_size = 0.001
# thres = 10**(-6)

img_test_raw = []
Test_result = []

def load_img(dir_path):
    global img_test_raw
    global Test
    img=[]
    img_test = []
    
    for i in range(class_num):
        train_random_list = random.sample(range(0,10),train_size)
        # train_random_list.sort()
        test_random_list = [ i for i in range(10) if i not in train_random_list ]
        print('test is',test_random_list)
        print(train_random_list)
  
        for k in range(class_sam):
            find = False
            for j in train_random_list:
                if (j == k):
                    # path = dir_path + '/s' + str(i+1) +'/'+str(j+1)+'.pgm'
                    path = osp.join(dir_path,'s'+str(i+1),str(j+1)+'.pgm')
                    print(path)
                    a = cv2.imread(path,0)
       
                    print('before',np.max(a))
                    a = a.flatten() / 255.0
                    print(np.max(a))
                    print(a.shape, type(a))
                    img.append(a)
                    find = True
                    break
            
            if not find:
                path = osp.join(dir_path,'s'+str(i+1),str(k+1)+'.pgm')
                # print(path)
                a = cv2.imread(path,0)
                img_test_raw.append(a)
                print('before',np.max(a))
                a = a.flatten() / 255.0
                print(np.max(a))
                print(a.shape, type(a))
                img_test.append(a)
        print('----------------')
        # print(img_test_raw[-1].shape)
    return img, img_test


def cal_softmax(x,w):
    a = np.dot(w,x.T)
    # print(a)
    # print(a.shape)
    # print('score\n',a)
    prob_unnorm = np.exp(a)
    prob_sum = np.array( np.sum(prob_unnorm,axis=0) )
    # print('The sum of',prob_sum)
    # print(prob_sum.shape)
    c_n, N = a.shape
    for i in range(c_n):
        for j in range(N):
            a[i][j] = prob_unnorm[i][j] / prob_sum[j]
            # print(i,j,a[i][j])
    print(a)
    return a


def cal_accuracy(w, t, x, test=False):
    global Test_result
    correct = 0
    total_N = x.shape[0]
    a = cal_softmax(x,w) 
    # print(a)
    result = np.where(a == np.amax(a, axis=0))

    if (test):
        Test_result = result[0]
    
    for index, c in enumerate(result[0]):
        if( t[index] == (c+1) ):
            correct+=1
    
    accuracy = correct / total_N
    return accuracy


def plot(error, accu, counter):
    fig = plt.figure('E(w)')
    fig.suptitle('E(w) curve',fontsize=25)
    plt.xlabel('iteration',fontdict={'fontsize':18})
    plt.ylabel('E(w)',fontdict={'fontsize':18})
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['figure.titlesize'] = 30
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.grid(True)
    plt.plot(range(counter),error, linestyle='-', zorder = 1, color = 'red', linewidth=1)
    plt.plot(range(counter),error,'ro', ms=4)

    fig2 = plt.figure('Accuracy')
    fig2.suptitle('Accuracy curve',fontsize=25)
    plt.xlabel('iteration',fontdict={'fontsize':18})
    plt.ylabel('Accu',fontdict={'fontsize':18})
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.rcParams['grid.alpha'] = 0.2
    plt.rcParams['figure.titlesize'] = 30
    plt.grid(True)
    plt.plot(range(counter),accu, linestyle='-', zorder = 1, color = 'blue', linewidth=1)
    plt.plot(range(counter),accu,'bo', ms=4)
    print('The error is:\n',error)
    print('The accuracy is:\n',accu)
    plt.show()



def do_gradient_descent(imgs, target, train = True):

    total_N = np.array(imgs).shape[0]
    w = np.zeros((class_num,len(imgs[0])))
    counter = 0
    error_record = []
    accu_record = []
    last_E = 0

    while (True):
        E = 0
        Gradient = np.zeros(w.shape)
        a = cal_softmax(np.array(imgs),w)
        print(np.max(imgs),np.max(w))

        for c in range(class_num):
            grad_e = 0
            for n in range(total_N):
                print('at counter',counter,c,n)
                print(a[c][n])

                if (math.isnan(a[c][n])):
                    print('NAN OCCURRED')
                    print('The learning rate is ',step_size)
                    print('The error is:\n',error_record)
                    print('The accuracy is\n',accu_record)
                    exit(-1)
            
                t = ( target[n] == (c+1) )
                error = t * np.log(a[c][n])
                grad_e += (a[c][n]-t)* imgs[n]
                
                E -= error
            Gradient[c] = grad_e
        error_record.append(E)
        w = w - step_size*Gradient
        # print('w is\n',w)
        print('Error is\n',E)
        counter += 1

        accuracy = cal_accuracy(w,target,np.array(imgs))
        accu_record.append(accuracy)
        
        # print('last_E is:',last_E)
        stop = ( math.fabs( ((error_record[-1]-last_E)/(last_E+0.0001))) < thres )


        # if ( stop  or counter >= itera_num or counter == 16):
        if ( stop  or counter >= itera_num ):
            print(last_E, E)
            # print(math.fabs((error_record[-1]-last_E)/(last_E+0.0001)))
            print(counter)

            
            plot(error_record, accu_record, counter)


            return w


        last_E = E



def show_result(target_test,accu_test):
    global img_test_raw
    print('In show')

    img_cut =[img_test_raw[i:i+5] for i in range(0,len(img_test_raw),5)]
    target = [Test_result[i:i+5] for i in range(0,len(Test_result), 5)]

    h,w = img_test_raw[0].shape
    p = 50
    indent = 5
    H = 5*h + indent*4 + p*2 + 20
    W = 5*w + indent*4 + p*2
    img=np.zeros((H,W),np.uint8)
    img.fill(100)
 
    # print(img_cut[-1])
    accu = 'Accuracy: '+str(accu_test)

    cv2.putText(img,accu, (int(W/2)-60,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for index,row in enumerate(img_cut):
        for i in range(len(row)):
            # img_t = cv2.resize( img_cut[index][i], (h,w) )
            img_t = img_cut[index][i]
            # print(img_t.shape)
            cv2.putText(img_t, 'c:%d'%(target[index][i]+1), (60,15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            img[20+p+h*index+indent*index:20+p+h*index+indent*index+h,(i*w+i*5)+50:((i+1)*w+i*5)+50] = img_t

    cv2.imwrite('result.jpg',img)
    cv2.imshow('test result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def do_Newton(imgs, target, d):
    # imgs = imgs_arr.tolist()
    # print(len(imgs))
    print('In New, type of imgs is:',type(imgs))
    print(type(imgs[0]),imgs[0].shape)
    total_N = np.array(imgs).shape[0]
    w = np.zeros((class_num,len(imgs[0])))

    counter = 0
    error_record = []
    accu_record = []
    w_record = [] ######
    last_E = 0

    
    #calculate for gradient (class x data#)
    while (True):
        E = 0
        Gradient = np.zeros(w.shape)
        a = cal_softmax(np.array(imgs),w)
        print(np.max(imgs),np.max(w))

        for c in range(class_num):
            grad_e = 0
            for n in range(total_N):
                print('at counter',counter,c,n)
                print(a[c][n])

                if (math.isnan(a[c][n])):
                    print('NAN OCCURRED')
                    exit(-1)
            
                t = ( target[n] == (c+1) )
                error = t * np.log(a[c][n])
                grad_e += (a[c][n]-t)* imgs[n]
                
                E -= error
            Gradient[c] = grad_e
        error_record.append(E)
        w_record.append(w)#####
        


        ##cal hessian
        I = np.identity(d)
        Hj = []

        for k in range(class_num):
            Hessian = np.zeros((d,d))
            for n in range(total_N):
                out_product = np.dot( np.reshape( imgs[n],(len(imgs[n]),1) ) ,np.reshape( imgs[n],(len(imgs[n]),1) ).T)
                v = a[k][n]*(1-a[k][n])
                Hessian += (v * out_product)
                print('Outproduct for %d is:\n'%(n))
                # print(out_product)
            Hj.append(Hessian)
        for k in range(class_num):
            w[k] = w[k] - np.dot( Gradient[k] ,np.linalg.pinv(Hj[k]) ) 
        # print(Hessian, Hessian.shape)
        print('Our Hessian for different para',Hj)
     
        # # print('w is\n',w)
        print('Error is\n',E)
        print(last_E, E)
        counter += 1

        accuracy = cal_accuracy(w,target,np.array(imgs))
        accu_record.append(accuracy)

        # if ( stop  or counter >= itera_num ):'
        if ( counter >= itera_num ):
            print(last_E, E)
            # print(math.fabs((error_record[-1]-last_E)/(last_E+0.0001)))
            print('Stop @ ',counter)
            
            plot(error_record, accu_record, counter)

            return w


        last_E = E


# input whole_data (np.array(dxN)) output dimension-reduced data y (np.array(kxN)) 
def do_pca(whole_data, k):
    
    # whole_data = np.array(imgs + imgs_test).T
    # print(whole_data.shape)
    N = whole_data.shape[1] #num of date
    d = whole_data.shape[0] #num of parameter
    mean = np.zeros((N,1))
    scatter = np.zeros((d,d))


    mean = np.sum( whole_data, axis=1 )/N #10304x1
    # print(mean, mean.shape)
    # print(whole_data[:,0].shape)

 
    for i in range(N):
        c = np.dot( (whole_data[:,i].reshape(d,1)-mean),(whole_data[:,i].reshape(d,1)-mean).T )
        print(c.shape)
        scatter += c    

    print(scatter, scatter.shape)

    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)


    ## get first k eigenvetor to form transform matrix
    transform_w = np.zeros((d,k))
    for i in range(k):
        transform_w[:,i] = eig_pairs[i][1]

    y = transform_w.T.dot(whole_data)
    return y


if __name__ == '__main__':
    imgs,imgs_test = load_img(data_path)
    # print(type(np.array(imgs)),np.array(imgs).shape)
    # print(type(np.array(imgs_test)),np.array(imgs_test).shape)

    #########################Gradient descent###############################
    # seting target value
    for i in range(class_num):
        for j in range(train_size):
            target.append(i+1)
            target_test.append(i+1)
        
    w = do_gradient_descent(imgs, target, True)
    accu_test = cal_accuracy(w, target_test, np.array(imgs_test), test=True)
    print('The accuracy of test data %f'%(accu_test))
    show_result(target_test,accu_test)
    ######################################################################


    exit(0)

    #########################Newton######################################### would cal up to 10 mins for pca <3
    #pca
    k = 5
   
    whole_data = np.array(imgs + imgs_test).T
    N = whole_data.shape[1]
    ####This is what you want
    y = do_pca(whole_data,k)


  
    print(y,y.shape)
    y_train = y[:,:25]
    y_test = y[:,25:N]
    print('after pca')
    print('train is',y_train.shape,y_train)
    print('test is',y_test.shape,y_test)

    for i in range(class_num):
        for j in range(train_size):
            target.append(i+1)
            target_test.append(i+1)
    print(target)
    print(target_test)

    #convet to list of array 
    y_train = (y_train.T).tolist()
    y_test = (y_test.T).tolist()
    for i in range(len(y_train)):
        y_train[i] = np.array(y_train[i])
        y_test[i] = np.array(y_test[i])
    print(type(y_train))   
    
    w = do_Newton(y_train, target, k)
    accu_test = cal_accuracy(w, target_test, np.array(y_test), test=True)
    print('The accuracy of test data %f'%(accu_test))
    show_result(target_test,accu_test)

