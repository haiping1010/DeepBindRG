import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import pandas as pd
# the random data
#f=open('out.csv', 'r')
#arr=f.readlines()

df=pd.read_csv('out.csv', sep = '  ',header = None)

x = df.iloc[:,0].values
y = df.iloc[:,1].values
print x
print y
