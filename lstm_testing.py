from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import preprocess

if __name__ == "__main__":
    print("--------------------------------------------------------")
    # preprocess.save_data()
    trainX, testX, trainY, testY = preprocess.getTrainTest("mfcc")
    batchSize, windowSize, imbedSize = preprocess.getBatchWindow(trainX)
    batchSize_test, windowSize_test, imbedSize_test = preprocess.getBatchWindow(testX)

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (99,12)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = 12))

    # Compile & train & predict
    ii = 0
    batch = 114
    while (ii+batch) <= trainX.shape[0]:
        print("current iteration:", ii/114)
        backend.clear_session()
        X_batch = trainX[ii:ii+batch]
        Y_batch = trainY[ii:ii+batch]
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(X_batch, Y_batch, epochs = 5, batch_size = batch)
        ii += batch
    # backend.clear_session()
    # model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    # model.fit(np.array(trainX), np.array(trainY), epochs = 5, batch_size = 100)

    predicted = model.predict(testX)
    print("=== prediction ===")
    correct = 0
    for i in range(predicted.shape[0]):
        max_pred = 0
        max_index = 0
        for j in range(predicted.shape[1]):
            if predicted[i][j] > max_pred:
                max_pred = predicted[i][j]
                max_index = j
        if max_index == testY[i]:
            correct += 1
        # print("test", i, ":", max_index)
    print("correct: ", correct)
    print("accuracy:", correct/600)