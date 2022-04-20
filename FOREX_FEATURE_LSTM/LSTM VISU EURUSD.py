# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 10:49:37 2020

@author: yiming
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('C:/Users/yiming/Feature Eng/EURUSD_FULL_DATA.csv')
df.Date = pd.to_datetime(df.Date)
df = df.set_index('Date')
train,test = df['EURUSD=X'][:53],df['EURUSD=X'][53:]


"""
LSTM EURUSD=X
"""
import tensorflow as tf
from numpy import array
from numpy import hstack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

####Vanilla LSTM####

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
raw_seq = train
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=200, verbose=0)

X_test,y_test = split_sequence(test,n_steps)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
y_pred = model.predict(X_test)


plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[3:],y_test,label = 'reality')
plt.plot(test.index[3:],y_pred,color = 'r',label = 'prediction')
plt.title('Vanilla LSTM')
plt.legend()
plt.show()

####Stacked LSTM####
# split a univariate sequence
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = train
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=200, verbose=0)
# demonstrate prediction
X_test,y_test = split_sequence(test,n_steps)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
y_pred = model.predict(X_test)


plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[3:],y_test,label = 'reality')
plt.plot(test.index[3:],y_pred,color = 'r',label = 'prediction')
plt.title('Stacked LSTM')
plt.legend()
plt.show()


#########Bidirectional##########
# split a univariate sequence
from tensorflow.keras.layers import Bidirectional
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = train
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_features = 1
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
# define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=200, verbose=0)
# demonstrate prediction
X_test,y_test = split_sequence(test,n_steps)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
y_pred = model.predict(X_test)


plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[3:],y_test,label = 'reality')
plt.plot(test.index[3:],y_pred,color = 'r',label = 'prediction')
plt.title('Bidirectional LSTM')
plt.legend()
plt.show()

#####CONVLSTM####
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import Flatten
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = train
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# reshape from [samples, timesteps] into [samples, timesteps, rows, columns, features]
n_features = 1
n_seq = 2
n_steps = 2
X_train = X_train.reshape((X_train.shape[0], n_seq, 1, n_steps, n_features))
# define model
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=500, verbose=0)
# demonstrate prediction
X_test,y_test = split_sequence(test,4)
X_test = X_test.reshape((X_test.shape[0], n_seq, 1, n_steps, n_features))
y_pred = model.predict(X_test)


plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[4:],y_test,label = 'reality')
plt.plot(test.index[4:],y_pred,color = 'r',label = 'prediction')
plt.title('CONV LSTM')
plt.legend()
plt.show()



#########CNN LSTM#######
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = train
# choose a number of time steps
n_steps = 4
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=500, verbose=0)
# demonstrate prediction
X_test,y_test = split_sequence(test,4)
X_test = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))
y_pred = model.predict(X_test)


plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[4:],y_test,label = 'reality')
plt.plot(test.index[4:],y_pred,color = 'r',label = 'prediction')
plt.title('CNN LSTM')
plt.legend()
plt.show()

#################Multi steps##################
###Vector Output Model###
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = train
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=50, verbose=0)
# demonstrate prediction
X_test,y_test = split_sequence(test,n_steps_in, n_steps_out)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
y_pred = model.predict(X_test)


plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[4:],test[4:],label = 'reality')
plt.plot(test.index[4:],y_pred,color = 'r',label = 'prediction')
plt.title('3_steps_in,2_steps_out Vector Output LSTM')
plt.legend()
plt.show()



#####Encoder Decoder Model####
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
 
# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
raw_seq = train
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100, verbose=0)
# demonstrate prediction
X_test,y_test = split_sequence(test,n_steps_in, n_steps_out)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
y_pred = model.predict(X_test)
y_pred1 = []
y_pred2 = []
for i in range(len(y_pred)):
    y_pred1.append(y_pred[i][0])
    y_pred2.append(y_pred[i][1])

plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[4:],test[4:],label = 'reality')
plt.plot(test.index[4:],y_pred1,color = 'r',label = 'prediction')
plt.plot(test.index[4:],y_pred2,color = 'r',label = 'prediction')
plt.title('3_steps_in,2_steps_out Encoder Decoder LSTM')
plt.legend()
plt.show()

#####Multiple series input#####
#######with all features###############
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
in_seq1 = df['S&P'][:53].values
in_seq2 = df['STD'][:53].values
in_seq3 = df['Vol_global'][:53].values
in_seq4 = df['Vol_local'][:53].values
in_seq5 = df['Price_MA'][:53].values
in_seq6 = df['Primary risk'][:53].values
in_seq7 = df['FD'][:53].values
in_seq8 = df['Energy'][:53].values
in_seq9 = df['Correlation'][:53].values
in_seq10 = df['STD_short'][:53].values
in_seq11 = df['Price_LB'][:53].values
in_seq12 = df['Price_UB'][:53].values
in_seq13 = df['Price_LB_Price_UB_Distance'][:53].values
in_seq14 = df['Distance_MA'][:53].values
out_seq = train.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq10 = in_seq10.reshape((len(in_seq10), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq13 = in_seq13.reshape((len(in_seq13), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))

out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq4,in_seq5,in_seq6,in_seq7,in_seq8,in_seq9,in_seq10,in_seq11,in_seq12,in_seq13,in_seq14,out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
in_seq1 = df['S&P'][53:].values
in_seq2 = df['STD'][53:].values
in_seq3 = df['Vol_global'][53:].values
in_seq4 = df['Vol_local'][53:].values
in_seq5 = df['Price_MA'][53:].values
in_seq6 = df['Primary risk'][53:].values
in_seq7 = df['FD'][53:].values
in_seq8 = df['Energy'][53:].values
in_seq9 = df['Correlation'][53:].values
in_seq10 = df['STD_short'][53:].values
in_seq11 = df['Price_LB'][53:].values
in_seq12 = df['Price_UB'][53:].values
in_seq13 = df['Price_LB_Price_UB_Distance'][53:].values
in_seq14 = df['Distance_MA'][53:].values
out_seq = test.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq10 = in_seq10.reshape((len(in_seq10), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq13 = in_seq13.reshape((len(in_seq13), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq4,in_seq5,in_seq6,in_seq7,in_seq8,in_seq9,in_seq10,in_seq11,in_seq12,in_seq13,in_seq14,out_seq))
X_test, y_test = split_sequences(dataset, n_steps)
y_pred = model.predict(X_test)
plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[2:],test[2:],label = 'reality')
plt.plot(test.index[2:],y_pred,color = 'r',label = 'prediction')
plt.title('Multivariate LSTM with all features')
plt.legend()
plt.show()


############with selected features#############
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
in_seq1 = df['S&P'][:53].values
in_seq2 = df['STD'][:53].values
in_seq3 = df['Vol_global'][:53].values
in_seq5 = df['Price_MA'][:53].values
in_seq7 = df['FD'][:53].values
in_seq8 = df['Energy'][:53].values
in_seq9 = df['Correlation'][:53].values
in_seq11 = df['Price_LB'][:53].values
in_seq12 = df['Price_UB'][:53].values
in_seq14 = df['Distance_MA'][:53].values
out_seq = train.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))

out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq5,in_seq7,in_seq8,in_seq9,in_seq11,in_seq12,in_seq14,out_seq))
# choose a number of time steps
n_steps = 3
# convert into input/output
X, y = split_sequences(dataset, n_steps)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
in_seq1 = df['S&P'][53:].values
in_seq2 = df['STD'][53:].values
in_seq3 = df['Vol_global'][53:].values
in_seq5 = df['Price_MA'][53:].values
in_seq7 = df['FD'][53:].values
in_seq8 = df['Energy'][53:].values
in_seq9 = df['Correlation'][53:].values
in_seq11 = df['Price_LB'][53:].values
in_seq12 = df['Price_UB'][53:].values
in_seq14 = df['Distance_MA'][53:].values
out_seq = test.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq5,in_seq7,in_seq8,in_seq9,in_seq11,in_seq12,in_seq14,out_seq))
X_test, y_test = split_sequences(dataset, n_steps)
y_pred = model.predict(X_test)
plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[2:],test[2:],label = 'reality')
plt.plot(test.index[2:],y_pred,color = 'r',label = 'prediction')
plt.title('Multivariate LSTM with selected features')
plt.legend()
plt.show()



### multivariate multi-step stacked lstm###
 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
in_seq1 = df['S&P'][:53].values
in_seq2 = df['STD'][:53].values
in_seq3 = df['Vol_global'][:53].values
in_seq4 = df['Vol_local'][:53].values
in_seq5 = df['Price_MA'][:53].values
in_seq6 = df['Primary risk'][:53].values
in_seq7 = df['FD'][:53].values
in_seq8 = df['Energy'][:53].values
in_seq9 = df['Correlation'][:53].values
in_seq10 = df['STD_short'][:53].values
in_seq11 = df['Price_LB'][:53].values
in_seq12 = df['Price_UB'][:53].values
in_seq13 = df['Price_LB_Price_UB_Distance'][:53].values
in_seq14 = df['Distance_MA'][:53].values
out_seq = train.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq10 = in_seq10.reshape((len(in_seq10), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq13 = in_seq13.reshape((len(in_seq13), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))

out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq4,in_seq5,in_seq6,in_seq7,in_seq8,in_seq9,in_seq10,in_seq11,in_seq12,in_seq13,in_seq14,out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
in_seq1 = df['S&P'][53:].values
in_seq2 = df['STD'][53:].values
in_seq3 = df['Vol_global'][53:].values
in_seq4 = df['Vol_local'][53:].values
in_seq5 = df['Price_MA'][53:].values
in_seq6 = df['Primary risk'][53:].values
in_seq7 = df['FD'][53:].values
in_seq8 = df['Energy'][53:].values
in_seq9 = df['Correlation'][53:].values
in_seq10 = df['STD_short'][53:].values
in_seq11 = df['Price_LB'][53:].values
in_seq12 = df['Price_UB'][53:].values
in_seq13 = df['Price_LB_Price_UB_Distance'][53:].values
in_seq14 = df['Distance_MA'][53:].values
out_seq = test.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.reshape((len(in_seq6), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq10 = in_seq10.reshape((len(in_seq10), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq13 = in_seq13.reshape((len(in_seq13), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq4,in_seq5,in_seq6,in_seq7,in_seq8,in_seq9,in_seq10,in_seq11,in_seq12,in_seq13,in_seq14,out_seq))
X_test, y_test = split_sequences(dataset,3, 2)
y_pred = model.predict(X_test)
plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[3:],test[3:],label = 'reality')
plt.plot(test.index[3:],y_pred,color = 'r',label = 'prediction')
plt.title('3_steps_in,2_steps_out Multivariate LSTM with all features')
plt.legend()
plt.show()



#########with selected features###########
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
 
# define input sequence
in_seq1 = df['S&P'][:53].values
in_seq2 = df['STD'][:53].values
in_seq3 = df['Vol_global'][:53].values
in_seq5 = df['Price_MA'][:53].values
in_seq7 = df['FD'][:53].values
in_seq8 = df['Energy'][:53].values
in_seq9 = df['Correlation'][:53].values
in_seq11 = df['Price_LB'][:53].values
in_seq12 = df['Price_UB'][:53].values
in_seq14 = df['Distance_MA'][:53].values
out_seq = train.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))

out_seq = out_seq.reshape((len(out_seq), 1))
# horizontally stack columns
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq5,in_seq7,in_seq8,in_seq9,in_seq11,in_seq12,in_seq14,out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=200, verbose=0)
# demonstrate prediction
# demonstrate prediction
in_seq1 = df['S&P'][53:].values
in_seq2 = df['STD'][53:].values
in_seq3 = df['Vol_global'][53:].values
in_seq5 = df['Price_MA'][53:].values
in_seq7 = df['FD'][53:].values
in_seq8 = df['Energy'][53:].values
in_seq9 = df['Correlation'][53:].values
in_seq11 = df['Price_LB'][53:].values
in_seq12 = df['Price_UB'][53:].values
in_seq14 = df['Distance_MA'][53:].values
out_seq = test.values
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq5 = in_seq5.reshape((len(in_seq5), 1))
in_seq7 = in_seq7.reshape((len(in_seq7), 1))
in_seq8 = in_seq8.reshape((len(in_seq8), 1))
in_seq9 = in_seq9.reshape((len(in_seq9), 1))
in_seq11 = in_seq11.reshape((len(in_seq11), 1))
in_seq12 = in_seq12.reshape((len(in_seq12), 1))
in_seq14 = in_seq14.reshape((len(in_seq14), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2,in_seq3,in_seq5,in_seq7,in_seq8,in_seq9,in_seq11,in_seq12,in_seq14,out_seq))
X_test, y_test = split_sequences(dataset,3, 2)
y_pred = model.predict(X_test)
plt.figure(figsize = (20,5))
plt.ylabel("EURUSD=X")
plt.plot(test.index[3:],test[3:],label = 'reality')
plt.plot(test.index[3:],y_pred,color = 'r',label = 'prediction')
plt.title('3_steps_in,2_steps_out Multivariate LSTM with selected features')
plt.legend()
plt.show()