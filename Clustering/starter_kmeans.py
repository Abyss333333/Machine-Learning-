import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
# Distance function for K-means
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

def plot_sc(clusters,data,K):
  div = np.zeros(K)
  colors = ['red','green', 'blue', 'pink', 'yellow']
  for i in range (K):
    ci = np.equal(i,clusters)
    summation = np.sum(np.equal(i,clusters))
    result = np.format_float_positional(np.float16(summation * 100.0/len(clusters)))
    div[i] = result
    length = len(clusters)
    plt.scatter(data[clusters == i,0], data[clusters == i,1] ,c = colors[i], cmap =plt.get_cmap('Spectral_r'), label =f'{result} %', s = 25, alpha = 0.5)


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

  
  
  dimension = dim
  X = tf.placeholder("float", shape = [None,dimension])
  init_mean = tf.truncated_normal([K,dimension], stddev = 0.05)
  iterx =300
  
  loss_arr = np.empty(shape=[0], dtype = float)
  loss_arr_valid = []
  MU = tf.Variable(init_mean)

  pairwise_dis = distanceFunc(X,MU)
  red_min = tf.reduce_min(pairwise_dis, axis = 1)
  loss = tf.reduce_sum(red_min)
  adam_opt = tf.train.AdamOptimizer(learning_rate = 0.1, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-5).minimize(loss)

  global_var = tf.global_variables_initializer()
  s = tf.Session()
  s.run(global_var)

  for steps in range (iterx):
    centroid, lTrain ,_ = s.run([MU,loss,adam_opt], feed_dict= {X:data})
    
    loss_arr = np.append(loss_arr, lTrain)
    if is_valid:
      centroid_v, lVal ,_ = s.run([MU,loss,adam_opt], feed_dict={X:val_data})
      loss_arr_valid = np.append(loss_arr_valid, lVal)
   

  d_ = distanceFunc(X,MU)
  clu = tf.argmin(d_,1)
  clusters = s.run(clu, feed_dict= {X:data, MU:centroid})
  if is_valid:
    valid_clusters = s.run(clu, feed_dict= {X:val_data, MU:centroid_v})
  plot_sc(clusters,data,K)
    
 
  if is_valid:
    print("Loss Validation:", lVal)
    lval = np.format_float_positional(np.float32(lVal))
 

  
  fig = plt.figure(1)
  plt.title('K Means Clusters K = %i' %K)
  plt.legend( loc = "best")
  plt.ylabel('Y')
  plt.xlabel('X')
  plt.grid()
  if is_valid:
    fig.text(.1,.0005,f'Final Validation Loss: {lval}', ha = 'left')
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
  




