import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    result = np.maximum(x,0)
    return result

def softmax(x):
    ex = np.exp(x)
    denom_sum = np.sum(ex, axis = 1, keepdims = True)
    result = ex/denom_sum
    return result

def computeLayer(X, W, b):
    mult = np.matmul(X,W)
    result = mult + b
    return result 


def CE(target, prediction):
    
    log_pk = np.log(prediction)
    mult = np.multiply(target,log_pk)
    sum = np.sum(mult)
    result = - (sum/ (target.shape[0]))
    return result 


def gradCE(target, prediction):
    return (prediction - target)

def relu_der(i):
    i[i>0] = 1
    i[i<=0] = 0
    return i

def softmax_input(z):
    
    z = z - np.max(z, axis=1)[:,None]
    return z

def front_prop( Data, Wh, Bh, Wo, Bo, target, param):


    data_N = Data.shape[0]
    x = computeLayer(Data, Wh, Bh)
    o = relu(x)
    z= np.add(np.matmul(o, Wo), Bo)
    z = softmax_input(z)
    p = softmax(z)
    loss = CE(target, p)
    prediction = np.argmax(p, axis = 1)
    actual = np.argmax(target, axis = 1)
    check = np.equal(prediction, actual)
    accuracy = (np.sum((check == True)) / data_N)

    if param == 1:
        print("training Accuracy:" , accuracy)
    elif param == 2:
        print("Valid Accuracy:" , accuracy)
   

        
    return loss, accuracy



def b_prop(target,prediction, o, x, i, Wo, param):

    t_N = target.shape[0]
    p = gradCE(target, prediction)
    alpha = 0.0000001
    
    

    if param == 1: #  1.2.1, 1.2.2
        o_t = np.transpose(o)
        temp = np.ones((1,t_N))
        v_o = alpha * np.matmul(o_t, p)
        b_o = alpha * np.matmul(temp, p)
        return v_o, b_o
    
    if param == 2: # 1.2.3, 1.2.4
        i = relu_der(i)
        i_N = i.shape[0]
        x_t = np.transpose(x)
        temp = np.ones((1,i_N))
        Wo_t = np.transpose(Wo)
        mult1 = np.matmul (p, Wo_t)
        mult2 = mult1 * i
        mult3 = np.matmul (p, Wo_t)
        mult4 = mult3 * i
        v_h = alpha * np.matmul(x_t, mult2)
        b_h = alpha * np.matmul(temp, mult4)
        return v_h, b_h
  

    return 0

def values (Data, Wh, Bh, Wo, Bo):
    
    x = computeLayer(Data, Wh, Bh)
    o = relu(x)
    z= np.add(np.matmul(o, Wo), Bo)
    z = softmax_input(z)
    p = softmax(z)

    return x,o,p

def reshape (data):
    return data.reshape((data.shape[0], -1))





def learning (target, target_v,  trainData, validData, Wo,Wh,Vo,Vh, Bo,Bh):

    epochs = 200
    mommentum = 0.99
    training_accuracy = []
    valid_accuracy = []
    training_loss = []
    valid_loss = []
    vnew = Vo
    vnewB = Bo
    vnewh = Vh
    vnewhb = Bh
    

    for i in range(epochs):

        print(i)
        tr_loss, tr_accuracy = front_prop (trainData, Wh, Bh, Wo, Bo, target,0)
        va_loss, va_accuracy = front_prop (validData, Wh, Bh, Wo, Bo, target_v,0)
        training_loss.append(tr_loss)
        valid_loss.append(va_loss)
        training_accuracy.append(tr_accuracy)
        valid_accuracy.append(va_accuracy)
        

        x,o,prediction = values(trainData, Wh, Bh, Wo, Bo)

        v_o,b_o = b_prop (target, prediction,o, 0,0,0, 1 )
        vnew = mommentum * vnew + v_o
        Wo = Wo - vnew
        vnewB = mommentum * vnewB + b_o
        Bo = Bo - vnewB

        v_h, b_h = b_prop(target, prediction, 0,trainData,x,Wo,2)
        vnewh = mommentum * vnewh + v_h
        Wh = Wh - vnewh
        vnewhb = mommentum * vnewhb + b_h
        Bh = Bh - vnewhb

    
    return training_accuracy, valid_accuracy, training_loss, valid_loss, Wo, Bo, Wh, Bh

def Xavier_initialization (data):
    H = 1000
    mean = 0
    variance = 2.0/( H + 10)
    varianceH = 2.0/(data.shape[0] + H)
    Wo = np.random.normal(mean, np.sqrt(variance), (H, 10))
    Vo = np.full((H,10), 1e-5)
    Wh = np.random.normal (mean, np.sqrt(varianceH), (data.shape[1], H))
    Vh = np.full((data.shape[1], H), 1e-5)
    Bo = np.zeros((1,10))
    Bh = np.zeros((1,H))



    return Wo, Vo,Wh, Vh, Bo, Bh

if __name__ == '__main__':

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    trainData = reshape(trainData)
    validData = reshape(validData)
    
    ytr, yva, yte = convertOneHot(trainTarget, validTarget, testTarget)
    
    Wo, Vo, Wh, Vh, Bo, Bh = Xavier_initialization(trainData)

    training_accuracy, valid_accuracy,training_loss, valid_loss, W_o , B_o, W_h, B_h = learning (ytr, yva, trainData, validData, Wo,Wh,Vo,Vh,Bo,Bh)

    x_axis = range(200)
    plt.subplot(1,2,1)
    plt.plot(x_axis, training_loss)
    plt.plot(x_axis, valid_loss)
    plt.legend(['training loss', 'validation loss'])
    plt.subplot(1,2,2)
    plt.plot(x_axis, training_accuracy)
    plt.plot(x_axis, valid_accuracy)
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.suptitle('Training/Validation Loss and Accuracy ')

    plt.show()

    tl,ta = front_prop(trainData, W_h,B_h, W_o, B_o, ytr, 1 )
    vl,va = front_prop(validData, W_h,B_h, W_o, B_o, yva, 2 )
    





    























        


