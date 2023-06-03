from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
import preprocess

if __name__ == "__main__":
    print("--------------------------------------------------------")
    trainX, testX, trainY, testY = preprocess.getTrainTest("mfcc")
    batchSize, windowSize = preprocess.getBatchWindow(trainX)
    # trainX = np.asarray(np.array(trainX)).astype(np.float32)
    # trainY = np.asarray(trainY).astype(np.float32)
    trainX = K.cast_to_floatx(trainX) # tf.convert_to_tensor(trainX)
    trainY = K.cast_to_floatx(trainY) # tf.convert_to_tensor(trainY)
    # trainX = np.stack(trainX)
    print("X: ",np.array(trainX).dtype)

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, batch_input_shape = (trainX.shape[0], windowSize, 2)))
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
    hist = model.fit(np.array(trainX), np.array(trainY), epochs = 5, batch_size = batchSize)
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
