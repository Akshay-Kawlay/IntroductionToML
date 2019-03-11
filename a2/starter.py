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
    return np.maximum(0,x)

def gradReLU(x):
    return (x > 0)

def softmax(x):
    exp_x = np.exp(x-np.amax(x, axis=0))   #the negative term fixes NaN error when x gets very large
    exp_x_denom = np.sum(exp_x, axis=0)
    return exp_x/exp_x_denom

def gradSoftmax(x):
    e = np.ones((x.shape[1], x.shape[1]))    
    return np.multiply(np.dot(x,e.T), (1-np.dot(e,x.T)).T)

def computeLayer(X, W, b):
    return np.dot(W,X)+b

def CE(target, prediction):
    N = target.shape[1]
    ce = (-1/N)*np.sum(np.multiply(target,np.log(prediction)))
    return np.squeeze(ce)

def gradCE(target, prediction):         #Not sure where this is to be used or if it is required
    N = target.shape[1]
    return (-1/N)*np.multiply(target,(1/prediction))

def NN_forward(X, weights):
    ''' Forward Pass    
        returns the output prediction and intermediate network values for backpropagation
    '''
    W1 = weights["W1"]; W2 = weights["W2"]
    b1 = weights["b1"]; b2 = weights["b2"]
    
    S1 = computeLayer(X, W1, b1)
    X1 = relu(S1)
    S2 = computeLayer(X1, W2, b2)
    X2 = softmax(S2)
    store = {"S1": S1, "X1": X1, "S2":S2, "X2":X2}
    return store

def NN_backpropagation(X, Y, weights, momentum, store, lr, gamma):
    ''' '''
    N = X.shape[1]
    W1 = weights["W1"]; W2 = weights["W2"]
    b1 = weights["b1"]; b2 = weights["b2"]
    v1 = momentum["v1"]; v2 = momentum["v2"]
    vb1 = momentum["vb1"]; vb2 = momentum["vb2"]
    
    Yhat = store["X2"]
    X1 = store["X1"]
        
    #calculate gradients of error w.r.t weights
    #print(Yhat.shape)
    #print(X1.shape)
    #dE_dS2 = np.multiply(gradCE(Y, Yhat),gradSoftmax(Yhat))
    #print("dE_dS2 shape : ", dE_dS2.shape)
    dE_dS2 = (Yhat - Y)#*gradCE(Y, Yhat)
    dE_dW2 = (1/N)*np.dot(dE_dS2, X1.T)
    dE_db2 = (1/N)*np.sum(dE_dS2, axis=1, keepdims=True)  #try axis=0
    dE_dS1 = np.dot(W2.T, dE_dS2)*gradReLU(X1)
    dE_dW1 = (1/N)*np.dot(dE_dS1, X.T)
    dE_db1 = (1/N)*np.sum(dE_dS1, axis=1, keepdims=True)
    
    #update momentum
    v1 = gamma*v1 + lr*dE_dW1
    v2 = gamma*v2 + lr*dE_dW2
    vb1 = gamma*vb1 + lr*dE_db1
    vb2 = gamma*vb2 + lr*dE_db2
    
    #update weights
    W1 -= v1
    W2 -= v2
    b1 -= vb1
    b2 -= vb2
    
    weights = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    momentum = {"v1": v1, "vb1": vb1, "v2": v2, "vb2": vb2}
    
    return weights, momentum
   
def save_weights(weights):
    '''Store Weights during training'''
    W1 = weights["W1"]; W2 = weights["W2"]
    b1 = weights["b1"]; b2 = weights["b2"]
    
    np.savetxt('weights//W1.txt', W1)
    np.savetxt('weights//b1.txt', b1)
    np.savetxt('weights//W2.txt', W2)
    np.savetxt('weights//b2.txt', b2)
    print("weights saved")

def load_weights(X_train, Y_train, n_hidden):
    '''initialize or load weight parameters'''
    if os.path.isfile("weights/W1.txt"):
        W1 = np.loadtxt("weights/W1.txt")
    else:
        W1 = np.random.randn(n_hidden, X_train.shape[1])*(2/(n_hidden+X_train.shape[1]))        #Xavier initialization
    if os.path.isfile("weights/b1.txt"):
        b1 = np.loadtxt("weights/b1.txt")
        b1 = b1.reshape((b1.shape[0], 1))
    else:
        b1 = np.zeros((n_hidden, 1))
    if os.path.isfile("weights/W2.txt"):
        W2 = np.loadtxt("weights/W2.txt")
    else:
        W2 = np.random.randn(Y_train.shape[1], n_hidden)*(2/(Y_train.shape[1]+n_hidden))        #Xavier initialization
    if os.path.isfile("weights/b2.txt"):
        b2 = np.loadtxt("weights/b2.txt")
        b2 = b2.reshape((b2.shape[0], 1))
    else:
        b2 = np.zeros((Y_train.shape[1], 1))
    
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    

def NN_numpy(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs, n_hidden, lr, gamma):
    '''Neural Network Model with Numpy'''
    
    #initialize parameter weights
    weights = load_weights(X_train, Y_train, n_hidden)
    
    #momentum parameters
    v1 = np.full((n_hidden, X_train.shape[1]), 1e-5)
    vb1 = np.full((n_hidden, 1), 1e-5)
    v2 = np.full((Y_train.shape[1], n_hidden), 1e-5)
    vb2 = np.full((Y_train.shape[1], 1), 1e-5)
    
    momentum = {"v1": v1, "vb1": vb1, "v2": v2, "vb2": vb2}
    
    Y_train = Y_train.T     #being consistent with parameter dimensions
    X_train = X_train.T
    Y_valid = Y_valid.T     
    X_valid = X_valid.T
    Y_test = Y_test.T     
    X_test = X_test.T
    
    #For plotting error per iteration
    train_error = np.zeros((epochs,1))
    valid_error = np.zeros((epochs,1))
    test_error = np.zeros((epochs,1))
    train_acc = np.zeros((epochs,1))
    valid_acc = np.zeros((epochs,1))
    test_acc = np.zeros((epochs,1))
    
    #training
    for i in range(epochs):
        
        store = NN_forward(X_train, weights)
        Yhat = store["X2"]
        train_error[i] = CE(Y_train, Yhat)
        train_acc[i] = calculateAccuraccy(Y_train, Yhat)
        print("iteration : ",i, " CE = ", train_error[i])
        
        #validation
        valid_store = NN_forward(X_valid, weights); valid_Yhat = valid_store["X2"]
        valid_error[i] = CE(Y_valid, valid_Yhat)
        valid_acc[i] = calculateAccuraccy(Y_valid, valid_Yhat)
        
        #test
        test_store = NN_forward(X_test, weights); test_Yhat = test_store["X2"]
        test_error[i] = CE(Y_test, test_Yhat)
        test_acc[i] = calculateAccuraccy(Y_test, test_Yhat)
        
        if np.isnan(train_error[i]):
            break
        if (i+1)%25 == 0:
            save_weights(weights)
        weights, momentum = NN_backpropagation(X_train, Y_train, weights, momentum, store, lr, gamma)
  
    
    #plot
    print_errorCurve(train_error, valid_error, test_error)
    print_accCurve(train_acc, valid_acc, test_acc)
    
    #predict testData
    store = NN_forward(X_test, weights)
    acc = calculateAccuraccy(Y_test, store["X2"])
    print("Test Accuraccy = ", acc, "%")

def calculateAccuraccy(Y, Yhat):
    pred = np.argmax(Yhat, axis=0)
    Y_class = np.argmax(Y, axis=0)
    acc = np.sum(pred == Y_class)/Y.shape[1]
    return acc*100
    
def print_errorCurve(train_error, valid_error, test_error):
    ''' '''
    plt.plot(train_error, label="training error")
    plt.plot(valid_error, label="validation error")
    plt.plot(test_error, label="testing error")
    plt.legend(loc='upper right')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()

def print_accCurve(train_acc, valid_acc, test_acc):
    ''' '''
    plt.plot(train_acc, label="training accuracy")
    plt.plot(train_acc, label="validation accuracy")
    plt.plot(test_acc, label="testing accuracy")
    plt.legend(loc='lower right')
    plt.xlabel("Iterations")
    plt.ylabel("Accuraccy")
    plt.show()
    
def NN_tf(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs, n_hidden, lr):
    
    return None

def ReshapeData(X_train, X_valid, X_test):
    X_train = X_train.reshape(10000,784)
    X_valid = X_valid.reshape(6000,784)
    X_test = X_test.reshape(2724,784)
    return X_train, X_valid, X_test
    
def main():
    #get data
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test = loadData()
    
    #reshape data
    X_train, X_valid, X_test = ReshapeData(X_train, X_valid, X_test)
    Y_train, Y_valid, Y_test = convertOneHot(Y_train, Y_valid, Y_test)
    
    #check
    assert(X_train.shape == (10000, 784))
    assert(X_valid.shape == (6000, 784))
    assert(X_test.shape == (2724, 784))
    assert(Y_train.shape == (10000, 10))
    assert(Y_valid.shape == (6000, 10))
    assert(Y_test.shape == (2724, 10))
    
    '''PART1'''
    NN_numpy(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs=200, n_hidden=1000, lr=0.05, gamma=0.99)  #epochs=175, n_h=1000, lr=0.0025,gmma=0.99 => accuraccy=81.71%
    
    '''PART2'''
    NN_tf(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs=20, n_hidden=15, lr=0.01)
    
    
main()
