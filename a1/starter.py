import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time     #used to create random seed

X_valid = []
Y_valid = []

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        #np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
        #print(Data.shape)
        # print(trainTarget.shape)
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def getReshapedDatasets(trainData, validData, testData):
    X_train = trainData.reshape(3500,784)
    X_valid = validData.reshape(100,784)
    X_test = testData.reshape(145,784)
    
    return X_train, X_valid, X_test
    
def main():
    global X_valid, Y_valid
    
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    X_train, X_valid, X_test = getReshapedDatasets(trainData, validData, testData)
    Y_train, Y_valid, Y_test = trainTarget, validTarget, testTarget
        
    #X_train = np.insert(X_train, 0, 1, axis=1)     #to combine b in W
    # s = int(time.time()*1000%1000)  #comment this before submission to be consistent with marker seed
    # np.random.seed(s)
    d = len(X_train[0])
    W = np.zeros((d,1)) #np.random.rand(d,1)
    b = 0 #np.random.rand(1)
    reg = 0
    alpha = 0.0129
    iterations = 4000
    EPS = 0.001
    
    W_trained, b_trained = grad_descent(W, b, X_train, Y_train, alpha, iterations, reg, EPS, lossType="MSE")
    
    #calculate test error
    mse = MSE(W_trained,b_trained,X_test,Y_test, reg)
    print("MSE = ",mse)
    
    W_trained, b_trained = grad_descent(W, b, X_train, Y_train, alpha, iterations, reg, EPS, lossType="CE")
    
    #calculate test accuraccy
    acc = evaluate_logistic_model(W_trained,b_trained,X_test,Y_test)
    print("Classification Accuraccy = "+str(acc*100)+"%")

def evaluate_logistic_model(W_trained,b_trained,X_test,Y_test):
    '''
    Evaluates a trained model with test data
    '''
    Yhat = forward_propagation(W_trained, b_trained, X_test)
    # for i in range(len(Y_test)):
        # print(Yhat[i],Y_test[i])
    eval = Yhat > 0
    correct = eval==Y_test
    acc = sum(correct)/len(Y_test)
    
    return np.squeeze(acc)
    
    
def forward_propagation(W, b, x):
    '''
    One cycle of forward propagation.
    Returns a yhat prediction value used to calculate MSE
    '''
    #print(W.shape)
    yhat = np.dot(x,W)+b
    return yhat

def MSE(W, b, x, y, reg):
    # Your implementation here
    '''
    Mean squared error
    returns a float indicating total loss
    '''
    num = len(y)
    yhat = forward_propagation(W, b, x)
    ms_error = (1/(2*num))*(np.dot((yhat - y).transpose(), yhat - y )) + (0.5)*reg*np.sum(np.square(W)) 
    ms_error = np.squeeze(ms_error)   #removes redundant list brackets
    
    return ms_error

def gradMSE(W, b, x, y, reg):
    # Your implementation here
    '''
    function of MSE. Returns dW, db
    '''
    N = len(x)
    yhat = forward_propagation(W, b, x)
    dW = (1/N)*(np.dot(x.transpose(),yhat-y)) + (reg)*W
    db = (1/N)*np.sum(yhat-y)
    return dW, db

def sigm(t):
    val = (1.0/(1.0 + np.exp(-t)))  
    return val
    
def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    N = len(x)
    yhat = sigm(forward_propagation(W, b, x))
    #print(yhat)
    #print(np.dot((1-y).transpose(),np.log(1-yhat)))
    ce_error = (-1/N)*(np.dot(y.transpose(),np.log(yhat))+ np.dot((1-y).transpose(),np.log(1-yhat))) + (0.5)*reg*np.sum(np.square(W)) 
    ce_error = np.squeeze(ce_error)   #removes redundant list brackets
    return ce_error

def gradCE(W, b, x, y, reg):
    # Your implementation here
    N = len(x)
    yhat = sigm(forward_propagation(W, b, x))
    dW = (1/N)*np.dot(x.transpose(), yhat-y) + (reg/N)*W
    db = (1/N)*np.sum(yhat-y)
    
    return dW, db

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    # Your implementation here
    ''' 
    returns optimized weights and bias
    '''
    #store mse values to plot later
    error_t = np.zeros(iterations)
    error_v = np.zeros(iterations)
    
    for i in range(iterations):
        if lossType=="MSE":
            #print(i)
            e = MSE(W, b, trainingData, trainingLabels, reg)
            error_v[i] = MSE(W, b, X_valid, Y_valid, reg)
            error_t[i] = e
            dW, db = gradMSE(W, b, trainingData, trainingLabels, reg)
            
        elif lossType=="CE":
            e = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
            error_v[i] = crossEntropyLoss(W, b, X_valid, Y_valid, reg)
            error_t[i] = e
            #print(i)
            dW, db = gradCE(W, b, trainingData, trainingLabels, reg)
            
        if e < EPS:   #stop if error tolerance goal reached
            break
            
        W -= alpha*dW
        b -= alpha*db
        
    print(error_t[iterations-1])
    print(error_v[iterations-1])
    plot_errors(error_t[200:], error_v[200:])
    
    return W, b
    
def plot_errors(e_train, e_valid):
    plt.plot(e_train, label="training error")
    plt.plot(e_valid, label="validation error")
    plt.legend(loc='upper left')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
    
	
def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    tf.set_random_seed(421)
    iterations = 1000
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    X_train, X_valid, X_test = getReshapedDatasets(trainData, validData, testData)
    Y_train, Y_valid, Y_test = trainTarget, validTarget, testTarget
    N = len(X_train)
    d = len(X_train[0])
    rand_w = tf.random.truncated_normal((d,1),stddev=0.5,dtype=tf.float32)
    W = tf.Variable(rand_w, name="W", dtype=tf.float32)
    rand_b = tf.random.truncated_normal((1,1),stddev=0.5,dtype=tf.float32)
    b = tf.Variable(rand_b, name="b", dtype=tf.float32)  
    reg = tf.placeholder(tf.float32, name="lambda")
    X = tf.placeholder(tf.float32, shape=None)#(3500,784))
    Y = tf.placeholder(tf.float32, shape=None)#(3500,1))
    Yhat = tf.add(tf.matmul(X,W), b)
    error_t = np.zeros(iterations)
    error_v = np.zeros(iterations)
    test_error = 0
    batch_size = 1750
    
    if lossType == "MSE": 
        # Your implementation 
        print("MSE")
        error = tf.losses.mean_squared_error(Y, Yhat)   #tf.reduce_sum(tf.pow(Yhat - Y, 2)) / N #
        #optimizer =  tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error)        
        optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(error)
    elif lossType == "CE": 
        #Your implementation here
        print("CE")
        error = tf.losses.sigmoid_cross_entropy(Y, Yhat)
        #optimizer =  tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(error)
        optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(error)
        
    initialize = tf.global_variables_initializer()
    
    with tf.Session() as s:
        s.run(initialize)
        for i in range(iterations):
            print(i)
            if (i*batch_size)%N == 0:
                randIndx = np.arange(N)
                np.random.shuffle(randIndx)
                X_train, Y_train = X_train[randIndx], Y_train[randIndx]
            X_train_SGD = X_train[(i*batch_size)%N:(i+1)*batch_size%N]
            Y_train_SGD = Y_train[(i*batch_size)%N:(i+1)*batch_size%N]
            s.run(optimizer, feed_dict={X:X_train_SGD, Y:Y_train_SGD})
            error_t[i] = s.run(error, feed_dict={X:X_train, Y:Y_train})
            error_v[i] = s.run(error, feed_dict={X:X_valid, Y:Y_valid})
        #test error
        test_error = s.run(error, feed_dict={X:X_test, Y:Y_test})
    print("Final training error = ",error_t[iterations-1])
    print("Final validation error = ",error_v[iterations-1])
    if lossType=="CE":
        acc = evaluate_logistic_model(W,b,X_test,Y_test)
        print("Classification Accuraccy = "+str(acc*100)+"%")
    else:
        print("Final test error = ",test_error)
    plot_errors(error_t, error_v)
    
    
main()
#buildGraph(0.99, 0.9, 0.0001, "CE", 0.001)
#loadData()

