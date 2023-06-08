from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preprocess
import elevator
import os

if __name__ == "__main__":
    print("------LSTM with keras model------")
    # preprocess.save_data()
    featureType = "mfcc"
    trainX, testX, trainY, testY = preprocess.getTrainTest(featureType)
    batchSize, windowSize, imbedSize = preprocess.getBatchWindow(trainX)
    batchSize_test, windowSize_test, imbedSize_test = preprocess.getBatchWindow(testX)

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (windowSize,imbedSize)))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.25))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.25))

    # Adding the output layer
    model.add(Dense(units = 1))

    # Compile & train
    ii = 0
    batch = 100
    while (ii+batch) <= trainX.shape[0]:
        print("current iteration:", ii/batch)
        backend.clear_session()
        X_batch = trainX[ii:ii+batch]
        Y_batch = trainY[ii:ii+batch]
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(X_batch, Y_batch, epochs = 200, batch_size = batch)
        ii += batch

    # predict
    predicted = model.predict(testX)
    print(testX.shape)
    print(predicted.shape)
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
    print("correct: ", correct)
    print("accuracy:", correct/600)

    # connect to UI & real audio recording test
    img = elevator.openWindow()
    command = elevator.voiceDetect("test")
    for c in command:
        if featureType == "mfcc":
            audioFeature = preprocess.getMFCC(c)
            test = np.vstack((testX[0 : 599], np.expand_dims(audioFeature, axis=0)))
            predict = model.predict(test)

        elif featureType == "fbank":
            audioFeature = preprocess.getFbank(c)
            print(audioFeature.shape)
            test = np.vstack((testX[0 : 599], audioFeature))
            predict = model.predict(test)
            
        # print(predict.shape)
        max_index = 0
        for i in range(predict.shape[1]):
            if predict[599][i] > max_pred:
                max_pred = predicted[599][i]
                max_index = j
        if max_index in range(1, 10):
            elevator.floorChoose(img, max_index)
        elif max_index == 0:
            print("zero")
        elif max_index == 10:
            print("up")
        elif max_index == 11:
            print("down")
        
        os.remove(c)
    
    elevator.wait()
    elevator.closeWindow()
