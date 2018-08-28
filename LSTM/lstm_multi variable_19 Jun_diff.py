# -*- coding: utf-8 -*-
"""
Created on Wed May  2 13:36:35 2018

@author: ikira
"""
#%reset -f
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

np.random.seed(142)


os.getcwd()
os.chdir("F:\\meeta\\Sem Wiki Study Material\\M.Mgt\\Project\\DATA - USa")

# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(str(x), '%d-%m-%y')

series = read_csv('all_lstm.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series = series.drop(columns = ['Industry'], axis = 1)


std_dev = np.std(series)
mean_val = np.mean(series)

####################################################################################

# prepare data

n_lag = 96 #how many dependants (1,2,3,12)

n_batch = 1 # 11, 13

n_vars = 0
n_seq = 1 #how many timesteps to forecast
n_test = 12
n_epochs = 500

n_neurons = 100
#n_neurons_5 = 12
#n_neurons_2 = 12
#n_neurons_3 = 4
n_neurons_4 = 100

#optim = ['nadam','adadelta']
optim = 'adadelta'
#optim = 'rmsprop'
#optim = 'nadam'

timesteps = 1 #memory of the network

n = n_vars + n_lag


# convert time series into supervised learning problem
# extract raw values
raw_values = series.values


# transform into supervised learning problem X, y

supervised = series.dropna()
supervised = supervised.dropna(axis='columns')

# rescale values to -1, 1
supervised = (supervised - mean_val)/std_dev
d = supervised.isna()


supervised_values = supervised.values
# split into train and test sets
train, test = supervised_values[0:-n_test, :], supervised_values[-n_test:, :]
scaler = 0




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
    series_one = series.values[:,-1]
    
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        #invert scaling
        inv_scale = (forecast*std_dev)+mean_val
        #invert differencing
        index = len(series_one) - n_test + i - 1
        last_ob = series_one.values[index]
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




# fit an LSTM network to training data
def fit_lstm(train, n, n_seq, n_batch, nb_epoch, n_neurons):
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
#    model.add(Dropout(0.0001))
#    model.add(LSTM(n_neurons_5, return_sequences = True,
##                   unroll = True, 
#                   recurrent_activation = 'elu',
#                   activation = 'tanh',
#                   kernel_initializer = k.he_normal(seed=142),
#                   recurrent_initializer = k.he_normal(seed=142)
#                   ))
#    model.add(Dropout(0.01))
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
    model.add(Dropout(0.0001))
#    model.add(Dropout(0.01))
#    model.add(Activation('sigmoid'))
#    model.add(Dense(y.shape[1]))
    model.add(Dense(y.shape[1]))
    
    model.compile(loss='mean_squared_error', optimizer=optim)
	# fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
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
def make_forecasts(model, n_batch, test, n, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
   


#####################################################################################

# fit single model
model = fit_lstm(train, (n_lag+n_vars), n_seq, n_batch, n_epochs, n_neurons)




#####################################################################################

#n_batch_pred = 1
## re-define model
#X, y = train[:, 1:], train[:, 0]
#X = X.reshape(X.shape[0], 1, X.shape[1])
## design network
#new_model = Sequential()
#new_model.add(LSTM(n_neurons, 
#                   return_sequences = True, 
#                   batch_input_shape=(n_batch_pred, X.shape[1], X.shape[2]), 
#                   stateful=True,
#                   activation='relu', 
#                   recurrent_activation='hard_sigmoid'))
#new_model.add(LSTM(n_neurons_2, 
#                   stateful = True, 
#                   return_sequences = True,
#                   activation='tanh', 
#                   recurrent_activation='hard_sigmoid'))
##new_model.add(LSTM(n_neurons_3, stateful = True))
#new_model.add(LSTM(n_neurons_3, 
#                   stateful = True
#                   , 
#                   return_sequences = True,
#                   activation='tanh', 
#                   recurrent_activation='hard_sigmoid'
#                   ))
#new_model.add(LSTM(n_neurons_4, stateful = True,
#                   activation='tanh', recurrent_activation='hard_sigmoid'))
##new_model.add(Dropout(0.01))
##    model.add(Activation('sigmoid'))
#new_model.add(Dense(1))
#
#
## copy weights
#old_weights = model.get_weights()
#new_model.set_weights(old_weights)

# make forecasts
#forecasts = make_forecasts(new_model, n_batch_pred, test, (n_lag+n_vars), n_seq)
forecasts = make_forecasts(model, n_batch, test, (n_lag+n_vars), n_seq)

# inverse transform forecasts and test
#forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
#actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
#plot_forecasts(series, forecasts, n_test+2)


# plot just predictions
raw_values = series.values

model
#pyplot.plot(raw_values[-12:])
pyplot.plot(actual)
pyplot.plot(forecasts)
pyplot.show()






# summarize history for loss
#plt.plot(model.history['loss'])
#plt.plot(model.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper right')
#plt.show()


#####################################################################################

# fit repeats of the model to get an average rmse
# repeat experiment

#repeats = 30
#error_scores = list()
#for r in range(repeats):
#	# fit the model
#    lstm_model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
#	# forecast the entire training dataset to build up state for forecasting
#    train_reshaped = train[:, 0].reshape(len(train), 1, 1)
#    lstm_model.predict(train_reshaped, batch_size=1)
#	# walk-forward validation on the test data
#    predictions = list()
#        
#    for i in range(len(test)):
#        # make one-step forecast
#        X, y = test[i, 0:-1], test[i, -1]
#        # forecast
#        X = X.reshape(1, 1, len(X))
#        yhat = lstm_model.predict(X, batch_size=n_batch)
#        yhat = yhat[0,0]
#        # invert scaling
#        new_row = [x for x in X] + [yhat]
#        arra = numpy.array(new_row)
#        arra = arra.reshape(1, len(arra))
#        inverted = scaler.inverse_transform(arra)
#        inverted = inverted[0, -1]
#        # yhat = invert_scale(scaler, X, yhat)
#        # invert differencing
#        interval = len(test)+1-i
#        yhat = yhat + raw_values[-interval]
#        # yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
#        # store forecast
#        predictions.append(yhat)
#    # report performance
#    rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
#    print('%d) Test RMSE: %.3f' % (r+1, rmse))
#    error_scores.append(rmse)
#   
#
#
## summarize results
#results = DataFrame()
#results['rmse'] = error_scores
#print(results.describe())
#results.boxplot()
#pyplot.show()
