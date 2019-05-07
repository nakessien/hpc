#!/usr/bin/env python3.6
from math import sqrt
from numpy import split
from numpy import array, stack, mean, std
from pandas import read_csv, DataFrame
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout, RepeatVector, TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from keras.utils import plot_model
import pywt
from sys import argv


# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into standard weeks
    train, test = data[0:-1000], data[-1000:]
    # restructure into windows of weekly data
    train = array(split(train, len(train) / 1))
    test = array(split(test, len(test) / 1))
    return train, test


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores

# summarize scores


def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
    # flatten data
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)


def build_model(train, n_input, epochs, batch_size):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose = 2
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same',))
    model.add(Conv1D(filters=4, kernel_size=1, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add((LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0))
    model.add((LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0))
    model.add((LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0))
    model.add((LSTM(200, activation='relu', input_shape=(n_timesteps, n_features))))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    print(f"Training network with {epochs} epochs, {batch_size} batch size")
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    #plot_model(model, show_shapes=True, to_file='/Users/User/Downloads/2d_conv_lstm_autoencoder.png')
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, 0]
    # reshape into [1, n_input, 1]
    input_x = input_x.reshape((1, len(input_x), 1))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train, test, n_input,epochs,batch_size):
    # fit model
    model = build_model(train, n_input,epochs,batch_size)
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    predictions = list()
    for i in range(len(test)):
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    print(f"test.shape: {test.shape}")
    print(f"predictions.shape: {predictions.shape}")
    test1 = test.reshape((test.shape[0] * test.shape[1]), 1)
    YPred = predictions.reshape((predictions.shape[0] * predictions.shape[1], 1))
    YPred = sig * YPred + mu
    rmse = sqrt(mean(YPred - test1)**2)
    print('CNNLSTM1D RMSE: > %.3f' % rmse)
    YPred = array(YPred)
    # DataFrame(YPred).to_csv('/Users/nakessien/Downloads/pred.csv')
    YPred = YPred.reshape(YPred.shape[0], 1)
    ypred_vs_test = stack((YPred, test1))
    ypred_vs_test = ypred_vs_test.transpose()
    ypred_vs_test = ypred_vs_test.reshape(ypred_vs_test.shape[1], 2)
    return score, scores, ypred_vs_test, model


if __name__ == '__main__':
    # load the new file
    print('Welcome')
    dataset = read_csv(argv[1], header=0, index_col=0)
    epochs = int(argv[2])
    batch_size = int(argv[3])
    #extract wavelet
    #cA, cD = pywt.dwt(dataset, 'haar')
    #dataset = DataFrame(cA)
    # split into train and test
    train, test = split_dataset(dataset.values)
    print(f"values shape: {dataset.values.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train shape: {train.shape}")
    # normalize
    mu = mean(train)
    sig = std(train)
    train = (train - mu) / sig
    test = (test - mu) / sig
    # evaluate model and get scores
    n_input = 12
    score, scores, ypred_vs_test, model = evaluate_model(train, test, n_input,epochs,batch_size)
    # summarize scores
    summarize_scores('LSTM', score, scores)

    # saving scores vs days 
    days = [1]
    dfscores_day = stack((days, scores))
    dfscores_day_filename = "scores.csv"
    print(f"Writing {dfscores_day_filename}")
    DataFrame(dfscores_day).to_csv(dfscores_day_filename)

    # 
    ypred_vs_test_filename = os.path.join('RESULTS','Spred1-CNNLSTM_AE.csv')
    print(f"Writing {ypred_vs_test_filename}")
    DataFrame(ypred_vs_test).to_csv(ypred_vs_test_filename)
    model_filename = 'CNNLSTM.h5'
    print(f"Writing {model_filename}")
    model.save(model_filename)
 
    #pyplot.plot(days, scores, marker='o', label='CNNLSTM_AE')
    #pyplot.show()
