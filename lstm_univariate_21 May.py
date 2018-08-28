# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:36:35 2018

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
from keras.layers import initializers as k
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np
import os
import matplotlib.pyplot as plt

os.getcwd()
os.chdir("F:\\meeta\\Sem Wiki Study Material\\M.Mgt\\Project\\DATA - USa")

np.random.seed(142)

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(str(x), '%d-%m-%y')

series = read_csv('volumes.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#series = read_csv('volumes_diff_sea.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = series['Industry']

# scaling
min_val = min(series)
max_val = max(series)

std_dev = np.std(series)
mean_val = np.mean(series)

#series = (dataset - mean_val)/std_dev




# convert time series into supervised learning problem
def series_to_supervised(data, n_in, n_out=1, dropnan=True):
#    data = series
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
#    for i in range(n_in, 0, -1):
    for i in lags:
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)


#diff_series = difference(series, 1)
#diff_values = diff_series.values
#
#min_val = min(diff_series)
#max_val = max(diff_series)
#
#std_dev = np.std(diff_series)
#mean_val = np.mean(diff_series)


# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    
    # rescale values to -1, 1
    scaler = 0
    scaled_values = (diff_values - mean_val)/std_dev
#    scaler = MinMaxScaler(feature_range=(-1, 1))
#    scaled_values = scaler.fit_transform(diff_values)
#    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[:-n_test], supervised_values[-n_test:]
    return scaler, train, test



# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, n_epochs):
	# reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], timesteps, X.shape[1])
	# design network
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences = True, 
                   batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True,
#                   unroll = True, 
#                   recurrent_activation = 'elu',
#                   activation = 'tanh',
                   
#                   kernel_initializer = k.he_normal(seed=142),
#                   recurrent_initializer = k.he_normal(seed=142)
                   
#                   kernel_initializer = k.orthogonal(seed=142),
#                   recurrent_initializer = k.orthogonal(seed=142)
                   
                   kernel_initializer = k.glorot_normal(),
                   recurrent_initializer = k.glorot_normal()
                   
                   ))
#    model.add(Dropout(0.001))
#    model.add(LSTM(n_neurons_5, return_sequences = True,
##                   unroll = True, 
#                   recurrent_activation = 'elu',
#                   activation = 'tanh',
#                   kernel_initializer = k.he_normal(seed=142),
#                   recurrent_initializer = k.he_normal(seed=142)
#                   ))
#    model.add(LSTM(n_neurons_2, return_sequences = True,
##                   unroll = True, 
##                   recurrent_activation = 'elu',
##                   activation = 'tanh',
#                   kernel_initializer = k.he_normal(seed=142),
#                   recurrent_initializer = k.he_normal(seed=142)
#                   ))
#    model.add(LSTM(n_neurons_3, return_sequences = True,
##                   unroll = True, 
#                   recurrent_activation = 'elu',
#                   activation = 'tanh',
#                   kernel_initializer = k.he_normal(seed=142),
#                   recurrent_initializer = k.he_normal(seed=142)
#                   ))
    model.add(LSTM(n_neurons_4 
#                   ,
#                   unroll = True
                   ))

#    model.add(TimeDistributed( y.shape[1] ))
##        , name = "linear"))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer=optim)
    #adadelta, nadam, adam, sgd
    
	# fit network
    for i in range(n_epochs):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model



# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, timesteps, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model, n_batch, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts


# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
		# create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
		# invert scaling
        inv_scale = (forecast*std_dev)+mean_val
#        inv_scale = scaler.inverse_transform(forecast)
#        inv_scale = inv_scale[0, :]
		# invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
		# store
        inverted.append(float(inv_diff[0]))
#    inverted = inverted.reshape(1, len(inverted))
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
    return rmse

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()
   
   
   
   
####################################################################################

## prepare data
#lags = [1,2,3,4,6,12]
#lags = [2,3,4,6,10,12] 

#lags = [1,2,3,4,5,6]

##lags = [2,3,11,12]
##lags = [2,3,4,11]
##lags = [2,3,4,11,12]
##lags = [1,2,3,4,11,12]
##lags = [1,2,3,4,6,11,12]
##lags = [1,3,5,11,12]
#lags = [1,3,5,10,12]
#lags = [2,5,7,10,11,12]
#lags = [2,3,5,7,10,12]

#lags = [1,2,5,7,10,12]

#lags = [1,3,5,7,10,11,12]
#lags = [1,4,5,7,10,11,12]
#lags = [1,2,3,6,9,11,12]



#lags = [1,2,3,5,9,11,12]
#lags = [1,2,3,5,8,11,12]
#lags = [2,3,5,6,8,11,12]



#lags = [1,3,5,9,11,12]
#lags = [4,5,6,7,8,10,12]
#lags = [1,2,5,9,11,12]
#lags = [1,3,4,9,10,11,12]

#lags = [2,5,6,7,8]



#lags = [2,5,7,10,11,12]

lags = [1,2,3,5,9,11,12]
n_lag = 7 #how many dependants (1,2,3,6,12)
#n_lag = 5


n_seq = 1 #how many timesteps to forecast
n_test = 12

#n_epochs = 1000
n_epochs = 1500

n_batch = 1
#optim = ['nadam','adadelta']
optim = 'adadelta'
#optim = 'rmsprop'
#optim = 'nadam'

n_neurons = 7
#n_neurons_5 = 24
#n_neurons_2 = 24
#n_neurons_3 = 12
n_neurons_4 = 7

#n_neurons_4 = 24
timesteps = 1 #memory of the network

scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

raw_values = series.values

####################################################################################

#results 

# fit single model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs)


history = model

#X = np.array([-8.48978615	,-7.95629025	,-10.36957645	,-8.65144348	,-7.4161396	,
#-10.07271385	,-8.607868026])
#    
#
#    
#X = X.reshape(1, timesteps, len(X))
#
## make forecast
#forecast = model.predict(X, batch_size=n_batch)
#forecast










# make forecasts
forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)
actual = [row[n_lag:] for row in test]
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# plot just predictions
raw_values = series.values
pyplot.plot(actual)
pyplot.plot(forecasts)
pyplot.show()


# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test)
actual = inverse_transform(series, actual, scaler, n_test)

evaluate_forecasts(actual, forecasts, n_lag, n_seq)

pyplot.plot(actual)
pyplot.plot(forecasts)
pyplot.show()


pyplot.plot(raw_values[-n_test:])
pyplot.plot(forecasts)
pyplot.show()

#describe(model)


#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


forecasts
actual




f = list()
s = std_dev.values[n_lag]
m = mean_val.values[n_lag]
for i in range(len(forecasts)):
    print(i)
    f.append( (forecasts[i][0]*s)+m )
f







#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

## changing the batch size to catch seasonality
#
#lags = [1,3,5,10,11,12]
##lags = [1,3,4,10,11,12]
#n_lag = 6 #how many dependants (1,2,3,6,12)
#
#n_batch = 1 # 11, 13
#
#n_vars = 0
#n_seq = 1 #how many timesteps to forecast
#n_test = 12
#n_epochs = 1000
#n_neurons = 5
#n_neurons_2 = 10
#n_neurons_3 = 10
#n_neurons_4 = 5
#n = n_vars + n_lag
#timesteps = 3
#
#scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
#
#raw_values = series.values
#
######################################################################################
## fit single model
#model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
######################################################################################
#
#n_batch_pred = 1
## re-define model
#X, y = train[:, :-1], train[:, -1]
#X = X.reshape(X.shape[0], timesteps, X.shape[1])
## design network
#new_model = Sequential()
#new_model.add(LSTM(n_neurons, return_sequences = True, 
#                   batch_input_shape=(n_batch_pred, X.shape[1], X.shape[2]), stateful=True))
#new_model.add(LSTM(n_neurons_2, return_sequences = True))
#new_model.add(LSTM(n_neurons_3, return_sequences = True))
#new_model.add(LSTM(n_neurons_4))
#new_model.add(Dense(1))
#new_model.add(TimeDistributed(Dense(y.shape[1]) ))
##        , name = "linear"))
#
## copy weights
#old_weights = model.get_weights()
#new_model.set_weights(old_weights)
#
## make forecasts
#forecasts = make_forecasts(new_model, n_batch_pred, test, n_lag, n_seq)
#
## inverse transform forecasts and test
#forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
#actual = [row[n_lag:] for row in test]
#actual = inverse_transform(series, actual, scaler, n_test+2)
#raw_test = list( raw_values[156:] )
#
#dif = list()
#for i in range(0, 12):
#    dif[i] = raw_test[i] - actual[i]
#
## evaluate forecasts
#evaluate_forecasts(actual, forecasts, n_lag, n_seq)
## plot forecasts
##plot_forecasts(series, forecasts, n_test+2)
#
#
## plot just predictions
#raw_values = series.values
#
#model
#pyplot.plot(actual)
#pyplot.plot(forecasts)
#pyplot.show()







#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################

# fit repeats of the model to get an average rmse
# repeat experiment

#repeats = 15
#error_scores = list()
#for r in range(repeats):
#	# fit the model
#    lstm_model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs)
#	# forecast the entire training dataset to build up state for forecasting
#    train_reshaped = train[:,:n_lag].reshape(len(train), 1, 7)
#    lstm_model.predict(train_reshaped, batch_size=1)
#	# walk-forward validation on the test data
#    # make forecasts
#    forecasts = make_forecasts(lstm_model, n_batch, test, n_lag, n_seq)
#    actual = [row[n_lag:] for row in test]
#    # evaluate forecasts
#    rmse = evaluate_forecasts(actual, forecasts, n_lag, n_seq)
#    # report performance
##    print('%d) Test RMSE: %.3f' % (r+1, rmse))
#    error_scores.append(rmse)
#   
#error_scores.append(0.562089)
#
## summarize results
#results = DataFrame()
#results['rmse'] = error_scores
#print(results.describe())
#results.boxplot()
#pyplot.show()
