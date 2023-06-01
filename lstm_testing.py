from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import preprocess

if __name__ == "__main__":
    trainX, testX, trainY, testY = preprocess.getTrainTest("mfcc")
    batchSize, windowSize = preprocess.getBatchWindow(trainX)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (trainX.shape[1:])))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = 1))

    # Compile & train & predict
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    hist = model.fit(trainX, trainY, epochs = 100, batch_size = batchSize)
    # predicted = model.predict(np.array(testX))

    # plot
    historydf = pd.DataFrame(hist.history, index = hist.epoch)
    # plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = hist.history['loss'][-1]
    acc = hist.history['accuracy'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    plt.savefig("./Plots/test.png")
    plt.close()
