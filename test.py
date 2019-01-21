
# coding: utf-8

# In[2]:


import os
import tensorflow as tf
import random
from keras.models import load_model
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, Model
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import  MaxPool2D
from keras.layers import  Softmax, Dropout, Flatten
from keras.initializers import glorot_uniform
import pandas as pd
import numpy as np
import glob
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from pandas import DataFrame

np.set_printoptions(threshold=np.nan)


# In[3]:


def identity_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, epsilon = 1e-6, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, epsilon = 1e-6, name = bn_name_base + '2c')(X)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    
    return X


def convolutional_block(X, f, filters, stage, block, s = 2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X


    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, epsilon = 1e-6, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    

    X = Conv2D(F2, (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed = 0))(X)
    #X = BatchNormalization(axis = 3, epsilon = 1e-6, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed = 0))(X)
    #X = BatchNormalization(axis = 3, epsilon = 1e-6, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(F3, (1, 1), strides = (s, s), padding = 'valid', name = conv_name_base + "1", kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
    #X_shortcut = BatchNormalization(axis = 3, epsilon = 1e-6, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    
    return X



def ResNet50(input_shape = (1000, 124, 1)):
    
    X_input = Input(input_shape)

    
    X = ZeroPadding2D((3, 3))(X_input)
    
    X = Conv2D(64, (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    #X = BatchNormalization(axis = 3, epsilon = 1e-6, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [5, 5, 5], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [5, 5, 5], stage=2, block='b')
    X = identity_block(X, 3, [5, 5, 5], stage=2, block='c')
    X = Dropout(0.2)(X)

    X = convolutional_block(X, f = 3, filters = [128, 128, 32], stage = 3, block = 'a', s = 2)
    X = identity_block(X, f = 3, filters = [128, 128, 32], stage = 3, block = 'b')
    X = identity_block(X, f = 3, filters = [128, 128, 32], stage = 3, block = 'c')
    X = identity_block(X, f = 3, filters = [128, 128, 32], stage = 3, block = 'd')
    X = Dropout(0.2)(X)
    #X = convolutional_block(X, f = 3, filters = [256, 256, 24], stage = 4, block = 'a', s = 2)
    #X = identity_block(X, f = 3, filters = [256, 256, 24], stage = 4, block = 'b')
    #X = identity_block(X, f = 3, filters = [256, 256, 24], stage = 4, block = 'c')
    #X = identity_block(X, f = 3, filters = [256, 256, 24], stage = 4, block = 'd')
    #X = identity_block(X, f = 3, filters = [256, 256, 24], stage = 4, block = 'e')
    #X = identity_block(X, f = 3, filters = [256, 256, 24], stage = 4, block = 'f')

    #X = convolutional_block(X, f = 3, filters = [512, 512, 48], stage = 5, block = 'a', s = 2)
    #X = identity_block(X, f = 3, filters = [512, 512, 48], stage = 5, block = 'b')
    #X = identity_block(X, f = 3, filters = [512, 512, 48], stage = 5, block = 'c')

    X = AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid', name = 'avg_pool')(X)
    
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(1, activation='linear', name='fc', kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model



def loadSplit(path):
    t = np.loadtxt(path,dtype=np.str)
    t1= []        
    for i in range(len(t)):
        t1.append([int(x) for x in list(t[i])])               
    output = np.array(t1)
    return output

def aucJ(true_labels, predictions):
    
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr,tpr)

    return auc

def randomShuffle(X, Y):
    idx = [t for t in range(X.shape[0])]
    random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]
    print()
    print('-' * 36)
    print('dimension of X after synthesis:', X.shape)
    print('dimension of Y after synthesis', Y.shape)
    print('label after shuffle:', '\n', DataFrame(Y).head())
    print('-' * 36)
    return X, Y

def synData(X_0, Y_0, X_1, Y_1, time):

    X_0_syn = X_0
    Y_0_syn = Y_0
    for i in range(time - 1):
        X_0_syn = np.vstack( (X_0_syn, X_0) )
        Y_0_syn = np.hstack( (Y_0_syn, Y_0) )

    print('dimension of generation data of X', X_0_syn.shape)
    print('dimension of generation data of Y', Y_0_syn.shape)
    print('dimension of generation data of X with label of 1', X_1.shape)
    print('dimension of generation data of Y with label of 1', Y_1.shape)

    #synthesis dataset
    X_syn = np.vstack( (X_0_syn, X_1) )
    Y_syn = np.hstack( (Y_0_syn, Y_1) )

    print()
    print('dimension of X after combination', X_syn.shape)
    print('dimension of Y after combination', Y_syn.shape)
    print(DataFrame(Y_syn).head())

    #shuffle data
    X_syn, Y_syn = randomShuffle(X_syn, Y_syn)
    
    return X_syn, Y_syn

data_path = 'all_data/4a*_learn_aa1000_0.4.txt'
#label_path = 'all_data/*_label.dat'

data_num = len(glob.glob(data_path))
#label_num = len(glob.glob(label_path))

print("data_num", data_num)
#print("label_num", label_num)


col_size=124
row_size=1000
data_samples = np.zeros((data_num, row_size, col_size))


# In[4]:


def loadlabel(path):
    base=os.path.basename(path)
    newname='all_data/'+ base[:4]+ '_label.dat'
    t = np.loadtxt(newname,dtype=np.float)
    return t

data_labels=[None]*data_num

index=0
for name in glob.glob(data_path):
    #print(name)
    t2=loadSplit(name)
    t3=loadlabel(name)
    data_samples[index,:,:] = t2
    data_labels[index]=t3
    index=index+1

data_labels=np.array(data_labels)
print(data_samples.shape)
print(data_labels)

