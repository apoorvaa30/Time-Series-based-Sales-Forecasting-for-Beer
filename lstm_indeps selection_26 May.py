# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:59:34 2018

@author: ikira
"""

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot

from sklearn import linear_model

os.getcwd()
os.chdir("F:\\meeta\\Sem Wiki Study Material\\M.Mgt\\Project\\DATA - USa")

np.random.seed(142)

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(str(x), '%d-%m-%y')

series = read_csv('all_vars.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#series = series['Industry']

# scaling
def scaler(data):
#    for i in range(data.shape[1]):
#        min_val = min(data[:,i])
#        max_val = max(data[:,i])
#        data[:,i] = (data[:,i] - min_val)/(max_val-min_val)
        
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
#        std_dev = np.std(data[:,i])
#        mean_val = np.mean(data[:,i])
#        data[:,i] = (data[:,i] - mean_val)/std_dev
    return data


#new_data = LassoLars
    


#data = series

#################################################################################################
#################################################################################################

#################################### Recursive feature selection : indeps #######################

series = read_csv('all_vars.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# separate into input and output variables
array1 = series.values
X = array1[:,3:]
y = array1[:,0]
# perform feature selection
rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 20)
fit = rfe.fit(X, y)
# report selected features
print('Selected Features:')
names = series.columns.values[3:]
for i in range(len(fit.support_)):
	if fit.support_[i]:
		print(names[i])
# plot feature rank
names = series.columns.values[3:]
ticks = [i for i in range(len(names))]
pyplot.bar(ticks, fit.ranking_)
pyplot.xticks(ticks, names)
pyplot.show()


        

#################################### Recursive feature selection : lags #######################

#lags_series = read_csv('all_lags.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#
## separate into input and output variables
#array2 = lags_series.values
#X = array2[:,1:]
#y = array2[:,0]
## perform feature selection
#rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), 7)
#fit = rfe.fit(X, y)
## report selected features
#print('Selected Features:')
#names = lags_series.columns.values[1:]
#for i in range(len(fit.support_)):
#	if fit.support_[i]:
#		print(names[i])
## plot feature rank
#names = lags_series.columns.values[1:]
#ticks = [i for i in range(len(names))]
#pyplot.bar(ticks, fit.ranking_)
#pyplot.xticks(ticks, names)
#pyplot.show()






#################################################################################################
#################################################################################################

#################################################################################################

series = read_csv('all_vars.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = series.drop(columns = [
        'National Currency per US Dollar, Period Average',
        'National Currency per US Dollar, End of Period'])

names = series.columns.values

#data = scaler(series.values)
data = series.values

X = data[:,2:]
y = data[:,1]

reg = linear_model.LassoLars(alpha=1.0, fit_intercept=True, 
                             verbose=True, normalize=False, 
                             precompute='auto', 
                             max_iter=40, eps=2.2204460492503131e-1, 
                             copy_X=True, fit_path=False, positive=False).fit(X,y)
x = array(reg.coef_)
cols = list()
for i in range(0, 96):
    if x[:,i]==0:
        continue;
    else:
        print(i)
        g = names[i]
        print(g)
#        cols = cols.append(names[i])

X = data[:,2:]
y = data[:,1]
model = linear_model.Lasso(alpha=0.3, fit_intercept=True, normalize=True, 
      precompute=False, copy_X=True, max_iter=40, tol=0.01, 
      warm_start=False, positive=False, random_state=None, 
      selection='cyclic')
model.fit(X,y)
model.coef_ ==0

names = series.columns.values
names[52]
names[53]










  
      




        
