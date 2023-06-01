# this .py file is for testing lstm model, not for ASR.
# predict Tesla stock price

# to-do: 將一維資料輸入調整為二維資料

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import preprocess

# def window_data(data, window_size):
#     X = []
#     y = []
#     i = 0
#     while (i+window_size < len(data)):
#         X.append(data[i : i+window_size]) # X: train data per group
#         y.append(data[i+window_size])     # y: target
#         i = i + 1
#     assert len(X) == len(y) # check whether X and y have equal length
#     return X, y

if __name__ == "__main__":
    trainX, testX, trainY, testY = preprocess.getTrainTest("mfcc")
    batchSize, windowSize = preprocess.getBatchWindow(trainX)

    # data preprocessing
    # tesla_stocks = pd.read_csv('tesla_stocks.csv')
    # data_to_use = tesla_stocks['Close'].values
    # scaler = MinMaxScaler(feature_range = (0, 1))
    # scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))

    # window_size = 7

    # X, y = window_data(scaled_data, window_size)

    # divide some data to be testing data
    # X_train = np.array(X[:700])
    # y_train = np.array(y[:700])
    # X_test = np.array(X[700:])
    # y_test = np.array(y[700:])

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (trainX.shape[1], 1)))
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



    # plt.plot(scaled_data, color = 'red', label = 'Real')  # red: real price
    # plt.plot(predicted, color = 'blue', label = 'Predicted')  # blue: predicted price
    # plt.title('Prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Tesla Stock Price')
    # plt.legend(loc="best")
    # plt.savefig("./Plots/test.png")
    # plt.close()
