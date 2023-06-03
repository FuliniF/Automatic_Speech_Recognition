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
    # preprocess.save_data()
    trainX, testX, trainY, testY = preprocess.getTrainTest("mfcc")
    batchSize, windowSize, imbedSize = preprocess.getBatchWindow(trainX)
    batchSize_test, windowSize_test, imbedSize_test = preprocess.getBatchWindow(testX)

    # Initialising the RNN
    model = Sequential()

    # Adding the LSTM layers and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (trainX.shape[1],12)))
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
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    hist = model.fit(np.array(trainX), np.array(trainY), epochs = 5, batch_size = batchSize)
    # print("ok")
    # for i in range(len(testX[0]/100)):
    #     tmp_testX = testX[i:i+100]
    #     predicted = model.predict(tmp_testX)
    #     print("predicted:",predicted)
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
    print("accuracy:", correct/600)
    # plot
    # historydf = pd.DataFrame(hist.history, index = hist.epoch)
    # # plt.figure(figsize=(8, 6))
    # historydf.plot(ylim=(0, max(1, historydf.values.max())))
    # loss = hist.history['loss'][-1]
    # acc = hist.history['accuracy'][-1]
    # plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))
    # plt.savefig("./test.png")
    # plt.close()
