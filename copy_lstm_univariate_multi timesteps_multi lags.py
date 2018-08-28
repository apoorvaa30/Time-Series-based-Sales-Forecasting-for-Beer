# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:40:28 2018

@author: ikira
"""

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
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np
import os
import matplotlib.pyplot as plt

os.getcwd()
os.chdir("F:\\meeta\\Sem Wiki Study Material\\M.Mgt\\Project\\DATA - USa")

np.random.seed(143)
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(str(x), '%d-%m-%y')

series = read_csv('volumes.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
#series = series['Industry']



# convert time series into supervised learning problem
def series_to_supervised(data, n_in, n_out=1, dropnan=True):
#    data = series['Industry']
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # input sequence (t-n, ... t-1)
#    for i in range(n_in, 0, -1):
    for i in lags:
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    
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

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
    
#	raw_values = series.values
#	# transform data to be stationary
#	diff_series = difference(raw_values, 1)
#	diff_values = diff_series.values
#	diff_values = diff_values.reshape(len(diff_values), 1)
#	# rescale values to -1, 1
#	scaler = MinMaxScaler(feature_range=(-1, 1))
#	scaled_values = scaler.fit_transform(diff_values)
#	scaled_values = scaled_values.reshape(len(scaled_values), 1)
#	# transform into supervised learning problem X, y
#	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
#	supervised_values = supervised.values
#	# split into train and test sets
#	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    
    raw_values = series.values[:,0]
    others = series.values[:,1:]
    
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(diff_values, n_lag, n_seq)
    
    dim = series.shape[0] - supervised.shape[0] 
    df = np.array(others[dim:,:])
    scaled_df = scaler.fit_transform(df)
#    scaled_df = scaled_df.reshape(len(scaled_df), 1)
    
    df1 = np.array(supervised.values)
    
    supervised_values = np.hstack((df1, scaled_df))

    # split into train and test sets
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    
    return scaler, train, test



# fit an LSTM network to training data
def fit_lstm(train, n, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
    X, y = train[:, 1:], train[:, 0]
    X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
    
#    model = Sequential()
#    model.add(LSTM(n_neurons, return_sequences = True, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
#    model.add(LSTM(n_neurons_2, return_sequences = True))
#    model.add(LSTM(n_neurons_3, return_sequences = True))
#    model.add(LSTM(n_neurons_4))
#    model.add(Dense(y.shape[1]))
    
    model = Sequential()
    model.add(LSTM(n_neurons, return_sequences = True, 
                   batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True,
                   activation='relu', 
                   recurrent_activation='hard_sigmoid'))
    model.add(LSTM(n_neurons_2, stateful = True, return_sequences = True,
                   activation='tanh', recurrent_activation='hard_sigmoid'))
#    model.add(LSTM(n_neurons_3, stateful = True))
    model.add(LSTM(n_neurons_3, stateful = True, 
                   return_sequences = True,
                   activation='tanh', 
                   recurrent_activation='hard_sigmoid'))
    model.add(LSTM(n_neurons_4, stateful = True,
                   activation='tanh', recurrent_activation='hard_sigmoid'))
#    model.add(Dropout(0.01))
#    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    
    
    model.compile(loss='mean_squared_error', optimizer='nadam')
	# fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model



# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts(model, n_batch, test, n, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 1:], test[i, 0]
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
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        series_one = series.values[:,0]
        index = len(series) - n_test + i - 1 
        last_ob = series_one[index]
#        inv_diff = inverse_difference(last_ob, inv_scale)
        inv_diff = inverse_difference(last_ob, forecast)
        inv_diff = inv_diff[0]
        # store
        inverted.append(inv_diff)
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
   
   
   
   
####################################################################################

# prepare data
lags = [1,2,3,4,6,12]
n_lag = 6 #how many dependants (1,2,3,6,12)

n_var = 3
n = n_lag + n_var
n_seq = 1 #how many timesteps to forecast
n_test = 12
n_epochs = 1000
n_batch = 1
n_neurons = 5
n_neurons_2 = 10
n_neurons_3 = 10
n_neurons_4 = 5

scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

raw_values = series.values

#####################################################################################

# fit single model
model = fit_lstm(train, n, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, test, n, n_seq)
# inverse transform forecasts and test
#forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
#actual = inverse_transform(series, actual, scaler, n_test+2)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+2)


# plot just predictions
raw_values = series.values
pyplot.plot(actual)
pyplot.plot(forecasts)
pyplot.show()

model


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
