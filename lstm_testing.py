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
from sklearn.preprocessing import MinMaxScaler

def window_data(data, window_size):
    X = []
    y = []
    i = 0
    while (i+window_size < len(data)):
        X.append(data[i : i+window_size]) # X: train data per group
        y.append(data[i+window_size])     # y: target
        i = i + 1
    assert len(X) == len(y) # check whether X and y have equal length
    return X, y

if __name__ == "__main__":

    # data preprocessing
    tesla_stocks = pd.read_csv('tesla_stocks.csv')
    data_to_use = tesla_stocks['Close'].values
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))

    window_size = 7

    X, y = window_data(scaled_data, window_size)

    # divide some data to be testing data
    X_train = np.array(X[:700])
    y_train = np.array(y[:700])
    X_test = np.array(X[700:])
    y_test = np.array(y[700:])

    # Initialising the RNN
    regressor = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compile & train & predict
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    stock_price = regressor.predict(np.array(X))

    # plot
    plt.plot(scaled_data, color = 'red', label = 'Real Stock Price')  # red: real price
    plt.plot(stock_price, color = 'blue', label = 'Predicted Stock Price')  # blue: predicted price
    plt.title('Tesla Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Tesla Stock Price')
    plt.legend(loc="best")
    plt.savefig("./Plots/test.png")
    plt.show()
    plt.close()