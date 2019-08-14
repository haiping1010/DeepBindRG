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

#*************************************jupyter_notebook*****************************************




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


data_path = 'all_data/*_learn_aa1000_0.4.txt'
#label_path = 'all_data/*_label.dat'

data_num = len(glob.glob(data_path))
#label_num = len(glob.glob(label_path))

print("data_num", data_num)
#print("label_num", label_num)


col_size=125
row_size=1000
data_samples = np.zeros((data_num, row_size, col_size))
data_labels = []

def loadlabel(path):
    base=os.path.basename(path)
    newname='all_data/'+ base[:4]+ '_label.dat'
    t = np.loadtxt(newname,dtype=np.float)
    return t

all_name=[]
index=0
for name in glob.glob(data_path):
    #print(name)
    t2=loadSplit(name)
    #t3=loadlabel(name)
    data_samples[index,:,:] = t2
    #data_labels.append(t3)
    index=index+1
    all_name.append(name)

#data_labels=np.array(data_labels)
print(data_samples.shape)
#print(data_labels.shape)



#######added by zhanghaiping ####################data_samples, data_labels = randomShuffle(data_samples, data_labels)

'''
import h5py
with h5py.File('data_aa1000_n.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset1",  data=data_samples)

import h5py
with h5py.File('label_aa1000_n.h5', 'w') as hf:
    hf.create_dataset("name-of-dataset2",  data=data_labels)
import h5py
with h5py.File('data_aa1000_n.h5', 'r') as hf:
    data_samples = hf['name-of-dataset1'][:]

with h5py.File('label_aa1000_n.h5', 'r') as hf:
    data_labels = hf['name-of-dataset2'][:]
'''


val_data_samples = data_samples[:, :, :]
#val_data_labels = data_labels[:]



X_val = val_data_samples.reshape(val_data_samples.shape[0],  row_size, col_size, 1)
#Y_val = val_data_labels.reshape(val_data_labels.shape[0],  1)

	
#*************************************jupyter_notebook*****************************************

##history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose = 120, validation_data = (X_test, Y_test), callbacks = [learning_rate_reduction])
##model.save("model_resnet_n_epoch60_linear_drop50.h5")
#model = load_model('model_resnet.h5')

model=load_model("model_resnet_n_linear_drop50_epoch_20.h5")


from sklearn import metrics

y_pred = model.predict(X_val)

#print (Y_val[0:20], y_pred[0:20])
fw=open('out_file.csv','w')
for i in range(y_pred.shape[0]):
     fw.write( all_name[i]  +"  "+ str(y_pred[i][0]))
     fw.write('\n')
def coeff_determination(y_true, y_pred):
    SS_res =  np.sum(np.square( y_true-y_pred ))
    SS_tot = np.sum(np.square( y_true - np.mean(y_pred) ) )
    return ( 1 - SS_res/(SS_tot + np.finfo(float).eps) )
import math



