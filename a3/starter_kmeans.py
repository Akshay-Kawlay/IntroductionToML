import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter

#Class colors
K_COLOR_MAP = {0 : 'r',
               1 : 'g',
               2 : 'b',
               3 : 'm',
               4 : 'y' }


# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)
is_valid = True
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    N = X.shape[0]
    d = X.shape[1]
    K = MU.shape[0]
    
    X = tf.expand_dims(X, -1)
    X = tf.tile(X, tf.stack([1, 1, K]))

    MU = tf.expand_dims(MU, -1)
    MU = tf.tile(MU, tf.stack([1, 1, N]))
    MU = tf.transpose(MU, perm=[2, 1, 0])

    diff = tf.subtract(X, MU)
    square_diff = tf.square(diff)

    pair_dist = tf.reduce_sum(square_diff, 1)
    
    return pair_dist
    
def getLoss(pair_dist):
    '''Calculates the loss from the pairwise distance matrix (NxK)'''
    return tf.reduce_sum(tf.reduce_min(pair_dist, axis=1))

def findKDistribution(distribution, K):
    '''finds percentage of data for each MU'''
    #print(distribution)
    N = distribution.shape[0]
    Items_per_K = Counter(distribution)
    K_percent = np.zeros((K,1))
    for i in range(K):
        K_percent[i] = Items_per_K[i]*100/N
    print(K_percent)
    return K_percent
    
    
def Kmeans(epochs, lr, K):
    print(data.shape)
    print(val_data.shape)
    N = data.shape[0]
    d = data.shape[1]
    N_valid = val_data.shape[0]
    
    #initialize centroids
    MU = tf.get_variable(name="MU", shape=(K,d))
        
    X = tf.placeholder(tf.float32, shape=(N,d))     #initialize tensors and placeholders
    pair_dist = distanceFunc(X, MU)
    loss = getLoss(pair_dist)
    adam_op = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    
    #initializing tensors for validation 
    X_valid = tf.placeholder(tf.float32, shape=(N_valid,d))#input placeholder dimension of Nxd
    pair_dist_v = distanceFunc(X_valid, MU)
    loss_v = getLoss(pair_dist_v)
    
    init = tf.global_variables_initializer()
    
    #initialize loss per iteration
    train_loss_array = np.zeros((epochs,1))
    valid_loss_array = np.zeros((epochs,1))
    
    with tf.Session() as sess:
        sess.run(init) 
        i = 1
        while(i < epochs):
            sess.run(adam_op, feed_dict={X: data})
            train_loss_array[i] = sess.run(loss, feed_dict={X: data})/N
            valid_loss_array[i] = sess.run(loss_v, feed_dict={X_valid: val_data})/N_valid
            print("Iteration: "+str(i)+" loss: "+str(train_loss_array[i]))
            i+=1
        #print(sess.run(pair_dist, feed_dict={X: data}))
        printLossCurve(train_loss_array, valid_loss_array)
        print("")
        print("Final training loss : ", train_loss_array[epochs-1])
        print("Final validation loss : ", valid_loss_array[epochs-1])
        print("")
        distribution = tf.argmin(pair_dist, axis=1)
        belongs_to = sess.run(distribution, feed_dict={X: data})
        K_percent = findKDistribution(belongs_to, K)
        
        print("")
        print("Distribution")
        print("Red : ", K_percent[0], "%")
        #print("Green : ", K_percent[1], "%")
        #print("Blue : ", K_percent[2], "%")
        #print("Magenta : ", K_percent[3], "%")
        #print("Yellow : ", K_percent[4], "%")
        print("")
            
        print_scatter2Dplot(data, belongs_to, K)

def print_scatter2Dplot(data, belongs_to, K):
    '''prints scatter plot of dataset'''
    K_color = [K_COLOR_MAP[i] for i in belongs_to]
    plt.scatter(data[:,0], data[:,1], c=K_color)
    plt.show()
        
def printLossCurve(train_loss_array, valid_loss_array):
    plt.plot(train_loss_array, label="K-means train loss")
    plt.plot(valid_loss_array, label="K-means valid loss")
    plt.legend(loc='upper right')
    plt.xlabel("Iterations")
    plt.ylabel("loss")
    plt.show()

Kmeans(epochs=500, lr=0.01, K=3)
