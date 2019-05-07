from numpy import mean
from numpy import std
from numpy import dstack, genfromtxt, argmax, stack, array
from pandas import read_csv, DataFrame
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import Dropout, Flatten
from keras.layers import LSTM, ConvLSTM2D, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from pandas_ml import ConfusionMatrix
from keras import optimizers
import time
from sklearn.model_selection import train_test_split

# load a single file as a numpy array
def load_file(filepath):
    dataframe = genfromtxt(filepath, delimiter=',')
    return dataframe

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group
    # load all files as a single array
    filenames = list()
    # total Speed
    filenames += ['/state_' + group + '_x' + '.csv']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/state_' + group + '_y' + '.csv')
    return X, y



# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
        # load all train
      
    trainX, trainy = load_dataset_group('Train', '')
    # load all test
    testX, testy = load_dataset_group('Test', '')
    print(testX.shape, testy.shape)
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    #print(trainy)
    return trainX, trainy, testX, testy

# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    
    verbose, epochs, batch_size = 2, 10, 6
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    # reshape output into [samples, timesteps, features]
    trainy = trainy.reshape((trainy.shape[0], trainy.shape[1]))
    testy = testy.reshape((testy.shape[0], testy.shape[1]))
    # define model
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=15, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=8, padding='same'))
    model.add(Conv1D(filters=512, kernel_size=15, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=8, padding='same',))
    model.add(Conv1D(filters=1024, kernel_size=15, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=8, padding='same'))
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_timesteps, n_features))))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(200, activation='relu', return_sequences=False, input_shape=(n_timesteps, n_features))))
    #model.add((Dense(200, activation='relu')))
    model.add((Dense(n_outputs, activation='sigmoid')))
    opt = optimizers.Adam(lr=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    class_weight = {0: 1., 1: 3.}
    # fit network
    history = model.fit(trainX, trainy, shuffle=False, epochs=epochs, validation_split=0.1, class_weight=class_weight, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    # list all data in history
    #print(history.history.keys())
    # make predictions
    #trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # compute confusion matrix
    y_pred = argmax(testPredict, axis=1, out=None)
    # save prediction dataset
    #prediction = DataFrame(trainPredict).to_csv('train_predictLSTM.csv')
    DataFrame(testPredict).to_csv('./test_predictLSTM.csv')
        
    
    y_actu = argmax(testy, axis=1, out=None)
    y_pred = y_pred.reshape(y_pred.shape[0],)
    y_actu = y_actu.reshape(y_actu.shape[0],)
    df = stack((y_pred, y_actu))
    df = df.transpose()
    df = df.reshape(df.shape[0], 2)
    DataFrame(df).to_csv('./classification.csv')
        #y_pred = np.delete(y_pred, 1)
    print(y_actu.shape, y_pred.shape)
    cm = ConfusionMatrix(y_actu, y_pred)
    cm.print_stats()
    d = cm.stats()
    f1 = list(d.items())[17]
    f1 = f1[1]
    print(f1)
    return accuracy





# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    
    
    
# run an experiment
def run_experiment(repeats=10):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    # repeat experiment
    scores = list()
    tic = time.time()
    for r in range(repeats):
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        #print('>#%d: %.3f' % (r+1, score))
        #scores.append(score)
    # summarize results
    summarize_results(score)
    toc = time.time()
    totalTime = toc-tic
    print(totalTime)
# run the experiment
run_experiment(1)