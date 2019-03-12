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

def convolutional_layer(x, weights, biases, strides=1):
    x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases)
    return tf.nn.relu(x) 

def maxpooling_layer(x, k=2):#k stands for kernal => 2 means 2x2 pool matrix with stride 2
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def batch_normalization_layer(x):
    mean, variance = tf.nn.moments(x,[0])#axes = [0] this is just the mean and variance of a vector.
    offset = tf.Variable(tf.zeros([100]))
    scale = tf.Variable(tf.ones([100]))
    epsilon = 1e-3
    return tf.nn.batch_normalization(x,mean,variance,offset,scale,epsilon)

def fully_connected_layer(x,weights, biases):
    # Reshape output x to fit fully connected layer input
    fully_connected = tf.reshape(x, [-1, weights['wd1'].get_shape().as_list()[0]])
    fully_connected = tf.add(tf.matmul(fully_connected, weights['wd1']), biases['bd1'])
    fully_connected = tf.nn.relu(fully_connected)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    results = tf.add(tf.matmul(fully_connected, weights['out']), biases['out'])
    return result

def softmax_cross_entropy_layer(x,y):
    return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)

def adam_optimizer(cost):
    return tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


def NN_tf(data, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs, batchSize, lr, noOfImages):
    '''Neural Network Model with Numpy'''
    print("Starting part 2")
    
    #initialize parameter weights
    x = tf.placeholder("float", [None, 28,28,1])#input placeholder dimension of BatchSize x 784
    y = tf.placeholder("float", [None, 10])#label of training images

    weights = {
    'wc1': tf.get_variable('W0', shape=(4,4,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W1', shape=(4*4*32,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W2', shape=(32,noOfImages), initializer=tf.contrib.layers.xavier_initializer()), 
    }
    biases = {
        'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'bd1': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('B3', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
    }

    #2. A 4 × 4 convolutional layer, with 32 filters, using vertical and horizontal strides of 1. 
    #3. ReLU activation
    layer = convolutional_layer(x, weights['wc1'], biases['bc1'])

    #4. A batch normalization layer
    layer = batch_normalization_layer(layer)

    #5. A max 2 × 2 max pooling layer
    layer = maxpooling_layer(layer, k=2)

    #6. Fully connected layer
    #7. ReLU activation
    #8. Fully connected layer
    prediction = fully_connected_layer(layer,weights,biases)

    #9. Softmax output
    #10. Cross Entropy loss
    cost = softmax_cross_entropy_layer(prediction, y)

    #Adam optimizer
    adam_optimizer_val = adam_optimizer(cost)
    
    #prediction and accurracies
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initializing the variables weights and biases
    init = tf.global_variables_initializer()

    #start tensorflow session https://www.datacamp.com/community/tutorials/cnn-tensorflow-python
    train_X = data.train.images.reshape(-1, 28, 28, 1)
    test_X = data.test.images.reshape(-1,28,28,1)
    train_y = data.train.labels
    test_y = data.test.labels

    with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(epochs):
        for batch in range(len(train_X)//batchSize):
            batch_x = train_X[batch*batchSize:min((batch+1)*batchSize,len(train_X))]
            batch_y = train_y[batch*batchSize:min((batch+1)*batchSize,len(train_y))]    
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(adam_optimizer_val, feed_dict={x: batch_x,
                                                              y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: test_X,y : test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))

    summary_writer.close()

def ReshapeData(X_train, X_valid, X_test):
    X_train = X_train.reshape(10000,784)
    X_valid = X_valid.reshape(6000,784)
    X_test = X_test.reshape(2724,784)
    return X_train, X_valid, X_test
    
def main():
    #get data
    X_train, X_valid, X_test, Y_train, Y_valid, Y_test, data = loadData()
    
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
    NN_tf(data, X_train, Y_train, X_valid, Y_valid, X_test, Y_test, epochs=50, batchSize=50, lr=0.0001, noOfImages=10)

    
main()
