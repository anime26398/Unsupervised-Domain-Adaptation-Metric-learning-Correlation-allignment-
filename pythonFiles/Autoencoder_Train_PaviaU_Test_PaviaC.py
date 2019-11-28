from __future__ import division, print_function, absolute_import
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import numpy
import PIL
from PIL import Image
np.random.seed(1337)  # for reproducibility
from math import sqrt

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import scipy.io
mat = scipy.io.loadmat('/home/aniruddha/deep-learning-projects/Siamese_Networks/Dataset/PaviaCentre.mat')
arr = mat['pavia']
arr = np.array(arr)
print(arr.shape)

import scipy.io
mat = scipy.io.loadmat('/home/aniruddha/deep-learning-projects/Siamese_Networks/Dataset/PaviaCentre_gt.mat')
arr1 = mat['pavia_gt']
arr1 = np.array(arr1)
print(arr1.shape)

a=[]
label=[]
k=0
for i in range(0,arr1.shape[0]):
    for j in range(0,arr1[i].shape[0]):
        a.append(arr[i][j])
        label.append(arr1[i][j])
        
a=np.array(a)
label=np.array(label)

X_train=[]
y_train=[]
for i in range (0,a.shape[0]):
    if(label[i]==2):
        y_train.append(0)
    if(label[i]==3):
        y_train.append(1)
    if(label[i]==4):
        y_train.append(2)
    if(label[i]==5):
        y_train.append(3)
    if(label[i]==7):
        y_train.append(4)
    if(label[i]==8):
        y_train.append(5)
    if(label[i]==9):
        y_train.append(6)
    if (label[i]==2 or label[i]==3 or label[i]==4 or label[i]==5 or label[i]==7 or label[i]==8 or label[i]==9):
        X_train.append(a[i])
X_train=np.array(X_train)
y_train=np.array(y_train)
print(X_train.shape)
print(y_train.shape)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train, random_state = 0)

from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
from sklearn.decomposition import PCA
pca = PCA(n_components=64)
X_train = pca.fit_transform(X_train)
print(X_train.shape)

import scipy.io
mat = scipy.io.loadmat('/home/aniruddha/deep-learning-projects/Siamese_Networks/Dataset/PaviaU.mat')
arr = mat['paviaU']
arr = np.array(arr)

import scipy.io
mat = scipy.io.loadmat('/home/aniruddha/deep-learning-projects/Siamese_Networks/Dataset/PaviaU_gt.mat')
arr1 = mat['paviaU_gt']
arr1 = np.array(arr1)
print(arr1.shape)

a=[]
label=[]
k=0
for i in range(0,arr1.shape[0]):
    for j in range(0,arr1[i].shape[0]):
        a.append(arr[i][j])
        label.append(arr1[i][j])
        
a=np.array(a)
label=np.array(label)
print(a.shape)
print(label.shape)

X_train1=[]
y_train1=[]
for i in range (0,a.shape[0]):
    if(label[i]==4):
        y_train1.append(0)
    if(label[i]==1):
        y_train1.append(1)
    if(label[i]==8):
        y_train1.append(2)
    if(label[i]==7):
        y_train1.append(3)
    if(label[i]==9):
        y_train1.append(4)
    if(label[i]==2):
        y_train1.append(5)
    if(label[i]==6):
        y_train1.append(6)
    if (label[i]==4 or label[i]==1 or label[i]==8 or label[i]==7 or label[i]==9 or label[i]==2 or label[i]==6):
        X_train1.append(a[i])
X_train1=np.array(X_train1)
y_train1=np.array(y_train1)

from sklearn.utils import shuffle
X_train1, y_train1 = shuffle(X_train1, y_train1, random_state = 0)
from sklearn.preprocessing import StandardScaler
X_train1 = StandardScaler().fit_transform(X_train1)
from sklearn.decomposition import PCA
pca = PCA(n_components=64)
X_train1 = pca.fit_transform(X_train1)
print(X_train1.shape)

print(X_train.max())
print(X_train1.max())
X_train=X_train.astype('float32')
X_train1=X_train1.astype('float32')
X_train=X_train/100
X_train1=X_train1/100

X_test=X_train1[20000:39332,:]
y_test=y_train1[20000:39332]
X_train1=X_train1[0:20000,:]
y_train1=y_train1[0:20000]

print(X_train.shape)
print(X_train1.shape)
print(X_test.shape)

learning_rate = 0.01
num_steps = 20
batch_size = 20
total_numbers = 291
display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 32 # 1st layer num features
num_hidden_2 = 16 # 2nd layer num features (the latent dim)
num_input = 64 
num_classes = 7

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


weights = {
    'encoder_h1': tf.Variable(tf.random_uniform([num_input, num_hidden_1], minval=-4*np.sqrt(6.0/(num_input + num_hidden_1)), maxval=4*np.sqrt(6.0/(num_input + num_hidden_1)))),
    'encoder_h2': tf.Variable(tf.random_uniform([num_hidden_1, num_hidden_2], minval=-4*np.sqrt(6.0/(num_hidden_1 + num_hidden_2)), maxval=4*np.sqrt(6.0/(num_hidden_1 + num_hidden_2)))),
    'decoder_h1': tf.Variable(tf.random_uniform([num_hidden_2, num_hidden_1], minval=-4*np.sqrt(6.0/(num_hidden_1 + num_hidden_2)), maxval=4*np.sqrt(6.0/(num_hidden_1 + num_hidden_2)))),
    'decoder_h2': tf.Variable(tf.random_uniform([num_hidden_1, num_input], minval=-4*np.sqrt(6.0/(num_input + num_hidden_1)), maxval=4*np.sqrt(6.0/(num_input + num_hidden_1)))),
    'classifier1_h': tf.Variable(tf.random_uniform([num_hidden_2, 10], minval=-4*np.sqrt(6.0/(10 + num_hidden_2)), maxval=4*np.sqrt(6.0/(10 + num_hidden_2)))),
    'classifier_h': tf.Variable(tf.random_uniform([10, num_classes], minval=-4*np.sqrt(6.0/(10 + num_classes)), maxval=4*np.sqrt(6.0/(10 + num_classes)))),
}
biases = {
    'encoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1])/sqrt(num_hidden_1)),
    'encoder_b2': tf.Variable(tf.truncated_normal([num_hidden_2])/sqrt(num_hidden_2)),
    'decoder_b1': tf.Variable(tf.truncated_normal([num_hidden_1])/sqrt(num_hidden_1)),
    'decoder_b2': tf.Variable(tf.truncated_normal([num_input])/sqrt(num_hidden_2)),
    'classifier1_b': tf.Variable(tf.truncated_normal([10])/sqrt(10)),
    'classifier_b': tf.Variable(tf.truncated_normal([num_classes])/sqrt(num_classes)),
}



# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
classify1 = tf.nn.sigmoid(tf.add(tf.matmul(encoder_op, weights['classifier1_h']), biases['classifier1_b']))
label_pred = tf.nn.softmax(tf.add(tf.matmul(classify1, weights['classifier_h']), biases['classifier_b']))
y_clipped = tf.clip_by_value(label_pred, 1e-10, 0.9999999)


# Targets (Labels) are the input data.
y_true = X
label_true = Y

# Define loss and optimizer, minimize the squared error
loss_autoencoder = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(label_true * tf.log(y_clipped)
                         + (1 - label_true) * tf.log(1 - y_clipped), axis=1))
loss_total = loss_autoencoder+cross_entropy_loss

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_total)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

from keras.utils import np_utils
y_test11 = np_utils.to_categorical(y_test)
y_train11 = np_utils.to_categorical(y_train)
print(y_train11.shape)
print(y_test11.shape)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(label_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)
batch_size = 64
num_batch = 1139

# Training
for i in range(0,200):
    k = 0 
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    avg_cost = 0
    for j in (0,num_batch):
        batch_x = X_train[k:k+batch_size,:]
        batch_y = y_train11[k:k+batch_size,:]
        k += 64
        #print(j)

    # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss_total], feed_dict={X: batch_x, Y: batch_y})
        avg_cost += l / num_batch
    
    print("Epoch:", (i + 1), "cost =", "{:.8f}".format(avg_cost))
    print("Epoch:", (i + 1), "accuracy =", "{:.8f}".format(sess.run(accuracy, feed_dict={X: X_train, Y: y_train11})))

    # on 200 epoch

print(sess.run([accuracy], feed_dict={X: X_test, Y: y_test11}))