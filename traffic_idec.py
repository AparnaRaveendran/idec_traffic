#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 12:07:32 2022

@author: aparna
"""

import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras import preprocessing
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
import datetime


def inertia_method(Z,k_range):
    '''
    Plot inertia

    Parameters
    ----------
    Z : tensor; input.
    k_range : int;upper bound of k values.

    Returns
    -------
    None.

    '''
    inertias = []
    mapping= {}

    for k in k_range:
        # Building and fitting the model
        kmeans = KMeans(n_clusters=k).fit(Z)
        kmeans.fit(Z)
        inertias.append(kmeans.inertia_)
        mapping[k] = kmeans.inertia_
        
    #plot inertia
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()

def plot_latent(Z):
    """
    plot latent reprenttaion using t-SNE
    sensor 8, 5 days in a week

    Parameters
    ----------
    Z : tensor; input.

    Returns
    -------
    None.

    """
    L = tf.reshape(Z,(26,-1,4))
    #print(L.shape)
    for s in range(8,9,1):
        day1 = L[s][864:1152]
        day2 = L[s][1153:1441]
        day3 = L[s][1442:1730]
        day4 = L[s][1731:2019]
        day5 = L[s][2020:2308]
    #print(day1.shape,day2.shape,day3.shape,day4.shape,day5.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    cmap = plt.cm.jet
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (0.0, 0.0, 0.0, 1.0)
    cmaplist[-1] = (.5, .5, .5, 1.0)

    bounds = np.linspace(0, 24,10)
    cb_ticks = [0,5,10,15,20,24]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)

    for z_t in [day1,day2,day3,day4,day5]:
        tsne =  TSNE(init='random',random_state = 0,perplexity= 100,learning_rate='auto').fit_transform(z_t)
        c1 = tf.split(tsne[:,0],num_or_size_splits=12+12+8)
        c2 = tf.split(tsne[:,1],num_or_size_splits=12+12+8)
        #colors = cm.rainbow(np.linspace(0, 1,len(c2)))
        ax.scatter(c1,c2, cmap=cmap,norm=norm,s=8)


    plt.title(f"sensor {s}")
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax2, orientation='vertical',ticks =cb_ticks,spacing='proportional')
    plt.show()


def target_distribution(q):
    """
    

    Parameters
    ----------
    q : tensor; assignment probability

    Returns
    -------
    p : tensor; traget distribution.

    """
    #print("inside target distribution",q.shape,freq.shape)
    p = tf.math.divide(tf.square(q),tf.reduce_sum(q,axis =0))
    p =  tf.math.divide(p ,tf.reduce_sum(p,axis =0))
    #print("target dist",p)
    return p

def soft_assignment(z,mu):
    """
    

    Parameters
    ----------
    z : tensor; input prob.
    mu : tensor; centroids
    
    Returns
    -------
    tensor
        assignmnet probability

    """
    alpha = 1.0
    def _pairwise_euclidean_distance(a,b):
        """
        Computes pairwise distance between each pair of points in two sets
        a:nxd
        b : kxd
        output : nxk
        """ 
        #squared norms of each row
        norm_a = tf.reduce_sum(tf.square(a),1)  
        norm_b =  tf.reduce_sum(tf.square(b),1) 
        norm_a  = tf.reshape(norm_a , [-1, 1]) #as row vector
        norm_b  = tf.reshape(norm_b , [1, -1])  #as column  vector
        ab = tf.matmul(a,b,transpose_b = True)  
        return (tf.sqrt(tf.maximum(norm_a + 2*ab -norm_b, 0.0)))

    dist = _pairwise_euclidean_distance(z,mu)   
    q = 1.0 / (1.0 + dist**2/alpha)**((alpha+1.0)/2.0)
    q = (q/tf.reduce_sum(q,axis =1, keepdims = True))
    return q

def get_grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


class DenoisedAutoencoder(Model):
  def __init__(self, input_dim,latent_dim,drop,input_shape_):
    super(DenoisedAutoencoder, self).__init__(name = "Denoised_AutoEncoder")
    self.latent_dim = latent_dim 
    self.drop = drop
    self.kernel = tf.keras.initializers.GlorotNormal(seed=0)
    self.input_dim = input_dim
    self.encoder = tf.keras.Sequential(layers =[
      layers.Flatten(),
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[0], activation='relu',kernel_initializer=self.kernel),
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[1], activation='relu',kernel_initializer=self.kernel),
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[2], activation='relu',kernel_initializer=self.kernel),
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[3], activation='relu',kernel_initializer=self.kernel),
    ],name = "Encoder")
    self.decoder = tf.keras.Sequential(layers = [
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[4], activation='relu',kernel_initializer=self.kernel),
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[5], activation='relu',kernel_initializer=self.kernel),
      layers.Dropout(self.drop),
      layers.Dense(latent_dim[6], activation='relu',kernel_initializer=self.kernel),                                  
      layers.Dense(input_dim, activation='relu',kernel_initializer=self.kernel),
      layers.Reshape(input_shape_) 
    ],name = "Decoder")


  def call(self, X):
    encoded = self.encoder(X)
    #print("encoded ",encoded.shape)
    decoded = self.decoder(encoded)
    #print("input ",X.shape)
    #print("decoded ",decoded.shape)
    #mse = tf.keras.losses.MeanSquaredError()
    #self.add_loss(mse(X,decoded))
    return decoded



class ClusteringLayer(Layer):
    def __init__(self,n_clusters):
        super(ClusteringLayer,self).__init__(name = 'Clustering_layer')
        #self.initial_weights =cluster_means
        self.alpha = 1.0
        self.n_clusters =n_clusters

    def build(self,input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(self.n_clusters,input_dim),
            initializer="random_normal",
            trainable=True,
            regularizer=tf.keras.regularizers.l2(0.1),
            name= "kernel")
        self.built = True
        
        #self.w = tf.Variable(
            #initial_value=mu(shape=cluster_means.shape, dtype="float32"),
            #trainable=True,)

    def soft_assignment(self,Z):
        #print(self.w.shape,Z.shape,self.w.dtype,Z.dtype)
        def _pairwise_euclidean_distance(a,b):
            """
            Computes pairwise distance between each pair of points in two sets
            a : nxd
            b : kxd
            output : nxk
            """ 
            #squared norms of each row
            norm_a = tf.reduce_sum(tf.square(a),1)  
            norm_b =  tf.reduce_sum(tf.square(b),1) 
            norm_a  = tf.reshape(norm_a , [-1, 1]) #as row vector
            norm_b  = tf.reshape(norm_b , [1, -1])  #as column  vector
            ab = tf.matmul(a,b,transpose_b = True)  
            return (tf.sqrt(tf.maximum(norm_a + 2*ab -norm_b, 0.0)))

        dist = _pairwise_euclidean_distance(Z,self.w)   
        q = 1.0 / (1.0 + dist**2/self.alpha)**((self.alpha+1.0)/2.0)
        q = (q/tf.reduce_sum(q,axis =1, keepdims = True))
        return q         
         
    def call(self,inputs): 
        return self.soft_assignment(inputs)

class Autoencoder(Model):
  def __init__(self, input_dim,latent_dim,input_shape_):
    super(Autoencoder, self).__init__(name = "AutoEncoder")
    self.latent_dim = latent_dim
    self.input_dim = input_dim
    self.encoder = tf.keras.Sequential(layers =[
        layers.Input(shape=input_shape_,name = 'input layer'),
        layers.Flatten(name = 'Flatten'),
        layers.Dense(latent_dim[0], activation='relu',kernel_initializer='he_uniform',name ='Dense_0'),
        layers.Dense(latent_dim[1], activation='relu',kernel_initializer='he_uniform',name ='Dense_1'),
        layers.Dense(latent_dim[2], activation='relu',kernel_initializer='he_uniform',name ='Dense_2'),
        layers.Dense(latent_dim[3], activation='relu',kernel_initializer='he_uniform',name ='Dense_3'),
        ],name = "Encoder")
    self.decoder = tf.keras.Sequential(layers = [
        layers.Input(shape=(latent_dim[3]),name = 'input layer'),
        layers.Dense(latent_dim[4], activation='relu',kernel_initializer='he_uniform',name ='Dense_4'),
        layers.Dense(latent_dim[5], activation='relu',kernel_initializer='he_uniform',name ='Dense_5'),
        layers.Dense(latent_dim[6], activation='relu',kernel_initializer='he_uniform',name ='Dense_6'),                                  
        layers.Dense(input_dim, activation='relu',kernel_initializer='he_uniform',name ='Output'),
        layers.Reshape(input_shape_,name ='Reshape') 
        ],name = "Decoder")


  def call(self, X):
    encoded = self.encoder(X)
    print("encoded ",encoded.shape)
    decoded = self.decoder(encoded)
    print("input ",X.shape)
    print("decoded ",decoded.shape)
    #mse = tf.keras.losses.MeanSquaredError()
    #self.add_loss(mse(X,decoded))
    return decoded


   
    
if __name__=="__main__":
    
    #args
    
    
    #read input
    input = np.load("data/input_tensor.npy")
    print(input.shape)

    X =tf.reshape(input,(input.shape[0]*input.shape[1],input.shape[2],input.shape[3]))
    print("input to AE", X.shape)
    
    
    #pre-train
    latent_dim = [32, 32, 128, 4, 128, 32, 32]
    input_dim = 36 #(=12*3)
    drop = 0.2
    input_shape_ = (12,3)
    epochs =36
    de_autoencoder = DenoisedAutoencoder(input_dim,latent_dim,drop,input_shape_)
    
    de_autoencoder.compile(optimizer=tf.keras.optimizers.Adam(),loss = 'mse',metrics='accuracy')
    history = de_autoencoder.fit(x =X,y =X, batch_size =288,epochs=epochs)
    
    #list all data in history
    print(history.history.keys())
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
    # summarize history for accuarcy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
    #save_weights
    de_autoencoder.save_weights(filepath = f"ae_weights_{epochs}/")
    
    # get latent representation
    Z = de_autoencoder.encoder(X)
    #R = de_autoencoder.decoder(Z)
    print("latent rep",Z.shape)
    #print("reconstructed rep",R.shape)
    print(Z)

    #save latent representation
    np.save(f"Z_{epochs}.npy", Z, allow_pickle=False)
    
    
    
    #plot latent
    #plot_latent(Z)
    
    #inertia method to identify number of clusters
    #inertia_method(Z, k_range = range(1,12))
    
    
    #K-Means
    K =3

    # initialize cluster centers using KMeans
    kmeans = KMeans(n_clusters= K, random_state=1).fit(Z)
    cluster_centers = kmeans.cluster_centers_
    #cluster lables
    labels = kmeans.predict(Z)
    #cluster frequency
    _, freq = np.unique(labels,return_counts= True)
    
    print("freq",freq)
    
    
    #clustering
    
    latent_dim = [32, 32, 128, 4, 128, 32, 32]
    input_dim = 36 #(=12*3)
    input_shape_ = (12,3)
    autoencoder = Autoencoder(input_dim,latent_dim,input_shape_)
    clustering_layer = ClusteringLayer(K)
    clustering_layer.build(input_shape = (4,))
    
    autoencoder.compile(optimizer=tf.keras.optimizers.SGD())
    
    
    #set_weights
    autoencoder.encoder.set_weights(de_autoencoder.encoder.weights)
    autoencoder.decoder.set_weights(de_autoencoder.decoder.weights)
    clustering_layer.set_weights([cluster_centers])
    
    #initialization
    #cluster_centers = tf.convert_to_tensor(cluster_centers,dtype = tf.float32)
    mu = tf.Variable(initial_value= cluster_centers,trainable = True,validate_shape=True,dtype = tf.float32)
    q = soft_assignment(Z,mu)
    Q = tf.argmax(q,axis =1)
    p = target_distribution(q)
    
        
    #model
    inputs =  tf.keras.Input(shape=input_shape_,name = 'Input')
    z = autoencoder.encoder(inputs)
    q = clustering_layer(z)
    outputs = autoencoder.decoder(z)
    
    dec_model = tf.keras.Model(inputs=inputs, outputs=[outputs,q])
    dec_model.compile(optimizer ='sgd', loss =['mse','kld'])
    
    #tf.keras.utils.plot_model(dec_model, "dec_traffic_model.png", show_shapes=True,expand_nested= False)
    
    
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.SGD()
    # Instantiate a loss function.
    mse = tf.keras.losses.MeanSquaredError()
    kld = tf.keras.losses.KLDivergence()
    
    print(dec_model.trainable_weights[-1])
    #print(dec_model.weights[0])
    
    
    #fine tuning
    epochs =10
    T = 2
    lambda_ =0.05
    
    loss_history = {'ae_loss' : [],'clust_loss' :[]}
    
    ratio = int(X.shape[0]/288)
    #print(ratio)
    
    
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        if ((epoch+1)%T == 0):
            logits = dec_model(X[:ratio*288])
            Y = logits[0]
            q = logits[1] 
            print("updating target distribution...")
            p = target_distribution(q)
    
        batches = tf.split(X[:ratio*288],num_or_size_splits=ratio,axis =0)
    
        p_batches = tf.split(p[:ratio*288],num_or_size_splits=ratio,axis =0)
        
        batch_ae_loss = 0
        batch_clust_loss =0
        
    
        for batch,p_batch in zip(batches,p_batches):
                        
                
            with tf.GradientTape(persistent=True) as tape:
                logits = dec_model(batch)
                Y = logits[0]
                q = logits[1]
        
                ae_loss_value = mse(batch,Y)
                clust_loss_value = lambda_ * kld(p_batch,q)
                    
        
            batch_ae_loss += ae_loss_value
            batch_clust_loss += clust_loss_value
                
    
            ae_grads = tape.gradient(ae_loss_value, dec_model.trainable_weights[:-1])
            #print("ae_grads", ae_grads)
            optimizer.apply_gradients(zip(ae_grads, dec_model.trainable_weights[:-1]))
        
            clust_grads = tape.gradient(clust_loss_value, [dec_model.trainable_weights[-1]])
            #print("clust_grads",clust_grads)
            optimizer.apply_gradients(zip(clust_grads, [dec_model.trainable_weights[-1]]))
            
        loss_history['ae_loss'].append(batch_ae_loss)
        loss_history['clust_loss'].append(batch_clust_loss)

        

        
    #list all data in history
    print(loss_history.keys())
    
    # summarize history for loss
    plt.plot(range(epochs),loss_history['ae_loss'])
    plt.title('ae loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    
    # summarize history for accuarcy
    plt.plot(range(epochs),loss_history['clust_loss'])
    plt.title('clust loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    output = dec_model(X)
    print(output[1])
    Q = tf.argmax(q, axis =1)
    print(Q.numpy())
