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

import pandas as pd
df=pd.read_csv('all_energies.sort',  sep=' +', header=None)

values=-df.iloc[0:,1].values

basename=df.iloc[0:,0].values
vina=dict(zip(basename,values))

df2=pd.read_csv('affinity.txt',  sep=':', header=None)

affinity=df2.iloc[0:,1].values

name=df2.iloc[0:,0].values
experiment=dict(zip(name,affinity))


Y_pred_m=[]
Y_val_m=[]
name_m=[]
for i,value1 in   vina.items():
    for j,value2 in experiment.items():
        if i[0:4]==j[0:4]:
            print i,j 
            Y_pred_m.append(value1)
            Y_val_m.append(value2)
            name_m.append(i)

y_pred_m=np.array(Y_pred_m)
Y_val_m=np.array(Y_val_m)

fw=open('out.csv','w')
for i in range(Y_val_m.shape[0]):
     print (Y_val_m[i], y_pred_m[i])
     fw.write(str(Y_val_m[i]) +"  "+ str(y_pred_m[i]) )
     fw.write('\n')

fw=open('out_list.csv','w')
for i in range(Y_val_m.shape[0]):
     print (Y_val_m[i], y_pred_m[i])
     fw.write(str(name_m[i])+"  "+str(Y_val_m[i]) +"  "+ str(y_pred_m[i])+ "  " +str(abs(Y_val_m[i]-y_pred_m[i])) )
     fw.write('\n')


from scipy.stats import linregress

#print ('R square first:', linregress(np.reshape(Y_val_m,Y_val_m.shape[0]), np.reshape(y_pred_m,y_pred_m.shape[0])))

slope, intercept, r_value, p_value, std_err=linregress(np.reshape(Y_val_m,Y_val_m.shape[0]), np.reshape(y_pred_m,y_pred_m.shape[0]))


print( 'R_value: ',   r_value)

print( 'MAE: ',   metrics.mean_absolute_error(Y_val_m, y_pred_m))

print('MSE:',metrics.mean_squared_error(Y_val_m, y_pred_m))

print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_val_m, y_pred_m)))

##print ('R value', model.score(Y_test, y_pred ))

from scipy import stats
slope, intercept, r_value, p_value, std_err=stats.linregress(Y_val_m.flatten(),y_pred_m.flatten())
 

from math import sqrt

summary=np.sum(np.square(Y_val_m.flatten()-(y_pred_m.flatten()*slope+intercept)))

SD=sqrt(summary/(Y_val_m.shape[0]-1))
print ('SD:' , SD)





