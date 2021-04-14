import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
from collections import Counter



# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    
    xstar = tf.expand_dims(X,0)
    mustar = tf.expand_dims(MU,1)
    difference = tf.subtract(xstar,mustar)
    squared = tf.square(difference)
    distance = tf.reduce_sum(squared,2)
    return (tf.transpose(distance))

def log_GaussPDF(X, mu, sigma):
    
    distance = distanceFunc(X, mu)
    sigma = tf.squeeze(sigma)
    dimension = tf.to_float(tf.rank(X))
    E = distance / (2*sigma)
    result = -0.5*dimension * (tf.log(2*np.pi * sigma))
    pdf = result - E
    return pdf

    

def log_posterior(log_PDF, log_pi):
    
    log_pi = tf.squeeze(log_pi)
    sum_  = tf.add(log_pi, log_PDF) 
    helper_sum = hlp.reduce_logsumexp(sum_ + log_pi, keep_dims=True)
    return (sum_ - helper_sum)

def plot_sc (data,num_pts, arr, K):
  clusters = Counter(arr)
      #valid_clusters = s.run(clu, feed_dict= {X:val_data, MU:centroid_v})
      
  arr = np.int32(arr)
  colors = ['red','green', 'blue', 'pink', 'yellow']
  for i in range (K):
        
    n = num_pts*(2/3)
    result = clusters[i]*100.0/n
    if result > 100:
      result = 100
        
    length = len(clusters)
    plt.scatter(data[arr == i,0], data[arr == i,1],cmap = plt.get_cmap('Set1') ,  label =f'{result:.2f} %', s = 25, alpha = 0.5)


def k_clusters (K,v):
    # Loading data
    data = np.load('data2D.npy')
    #data = np.load('data100D.npy')
    
    [num_pts, dim] = np.shape(data)

    #set is_valid to false
    is_valid = v
    # For Validation set
    if is_valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]

    
    np.random.seed(420)
    iterx =500
    loss_arr = []
    loss_arr_valid = []
    arr = []

    initpi = tf.Variable(tf.random_normal([K, 1], stddev = 0.05))
    lpi = tf.squeeze(hlp.logsoftmax(initpi))
    
    X = tf.placeholder("float", [None, dim], "X")
    init_mean = tf.random_normal([K,dim], stddev = 0.05)
    MU = tf.Variable(init_mean)
    init_sigma = tf.random_normal([K,1], stddev = 0.05)
    sigma = tf.exp(tf.Variable(init_sigma))
    pdf = log_GaussPDF(X, MU, sigma)

    

    red_min = hlp.reduce_logsumexp(pdf +lpi, 1, keep_dims=True)
    loss = - tf.reduce_sum(red_min)
    adam_opt = tf.train.AdamOptimizer(learning_rate = 0.1, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-5).minimize(loss)
    
    lpost = log_posterior(pdf, lpi)
    smax = tf.nn.softmax(lpost)
    relu = tf.argmax(smax,1)
    
    
 

   

  
    with tf.Session() as s:
      s.run(tf.global_variables_initializer())
      s.run(tf.local_variables_initializer())


      for steps  in range (iterx):
        _, lTrain, _, arr = s.run([MU,loss,adam_opt,relu], feed_dict= {X:data})
        loss_arr.append(lTrain)
        if is_valid:
          _, lVal ,_, _ = s.run([MU,loss,adam_opt,relu], feed_dict={X:val_data})
          loss_arr_valid.append(lVal)
   

      #d_ = distanceFunc(X,MU)

      plot_sc(data,num_pts, arr,K)
    
    
      
    

      if is_valid:
        lval = np.format_float_positional(np.float32(lVal))

  
      fig = plt.figure(1)
      plt.title('K Means Clusters K = %i' %K)
      plt.legend(loc = "best")
      plt.ylabel('Y')
      plt.xlabel('X')
      if is_valid:
        fig.text(.1,.0005,f'Final Validation Loss: {lval}', ha = 'left')
      plt.grid()
      plt.show()

    plt.figure(1)
    plt.plot(range(len(loss_arr)), loss_arr, c="g", label = "training Loss")
  
    plt.legend(loc = "best")
    plt.title('K Means')
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.show()

    return loss_arr_valid

if __name__ == "__main__":

  v = k_clusters(3,False)
  v = k_clusters(1,True)
  v = k_clusters(2,True)
  v = k_clusters(3,True)
  v = k_clusters(4,True)
  v = k_clusters(5,True)
  #v = k_clusters(5,True)
  #v = k_clusters(10,True)
  #v = k_clusters(15,True)
  #v = k_clusters(20,True)
  #v = k_clusters(30,True)

