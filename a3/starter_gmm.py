import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    
def MoG(epochs, lr, K):
    print(data.shape)
    print(val_data.shape)
    N = data.shape[0]
    d = data.shape[1]
    
    #initialize centroids
    MU = tf.get_variable(name="MU", shape=(K,d))
    phi = tf.get_variable(tf.float32, shape=(K,1))#variable dimension of Kx1
    psi = tf.get_variable(tf.float32, shape=(K,1))#variable dimension of Kx1
    #print(MU)
    #print(pi)
    #print(variance)
    
    #initialize tensors and placeholders
    X = tf.placeholder(tf.float32, shape=(N,d))#input placeholder dimension of Nxd
    variance = tf.exp(phi)
    log_GaussPDF_val = log_GaussPDF(X,MU,variance)
    log_pi_val = logsoftmax(psi)
    log_posterior_val = log_posterior(log_GaussPDF_val, log_pi_val)

    loss = getLoss(log_posterior_val, tf.squeeze(log_pi_val))

    adam_op = tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    
    init = tf.global_variables_initializer()
    
    #initialize loss per iteration
    loss_array = np.zeros((epochs,1))
    
    with tf.Session() as sess:
        sess.run(init) 
        i = 1
        while(i < epochs):
            sess.run(adam_op, feed_dict={X: data})
            loss_array[i] = sess.run(loss, feed_dict={X: data})
            print("Iteration: "+str(i)+" loss: "+str(loss_array[i]))
            i+=1
        #print(sess.run(pair_dist, feed_dict={X: data}))
        # distribution = tf.argmin(pair_dist, axis=1)
        # belongs_to = sess.run(distribution, feed_dict={X: data})
        # K_percent = findKDistribution(belongs_to, K)
        # print_scatter2Dplot(data, belongs_to, K)
        #printLossCurve(loss_array)
