import tensorflow.compat.v1 as tf
import preprocess
tf.disable_v2_behavior()
import numpy as np

# function "getCell"
"""
input:
- cell_size: the size of the LSTM cell
- keep_prob: the probability of keeping a neuron in the dropout layer

step:
1. create an LSTM cell of the specified size

note:
- dropout layer for preventing overfitting
"""
def getCell(cell_size, keep_prob):
    LSTM_cell = tf.nn.rnn_cell.LSTMCell(cell_size)
    droped_cell = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell, keep_prob)
    return LSTM_cell


# function "LSTMCell"
"""
input:
- numNeuron: the number of neurons in each layer of the LSTM cell
- X: input data to LSTM cell
- numLayer: the number of layers in the LSTM cell
- keep_prob: the probability of keeping a neuron in the dropout layer

step:
1. create an LSTM cell of the specified size
"""
def LSTMCell(numNeuron, X, numLayer, keep_prob):    
    cell = tf.nn.rnn_cell.MultiRNNCell([getCell(n, keep_prob) for n in numNeuron])
    init_state = cell.zero_state(tf.shape(X)[0], tf.float32)
    return cell, init_state


# function "outLayer"
"""
input:
- lstmOut: the output of the LSTM cell
- size: the number of output classes

step:
1. extract the final output of the LSTM cell (store in x)
2. create a densely connected layer where each neuron is connected to all the neurons in the previous layer
"""
def outLayer(lstmOut, size):
    x = lstmOut[:, -1, :]
    output = tf.layers.dense(x, size, activation = None)
    return output


# function "opt_loss"
"""
input:
- 
- 

step:
1. 
2. 
"""
def opt_loss(output, targets, learning_rate):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = targets), name = 'loss')
    optimizer = tf.train.AdamOptimizer(learning_rate)

    gradients = optimizer.compute_gradients(loss)
    #capped_gradients = [(tf.clip_by_value(grad, -clip, clip), var) for grad, var in gradients if grad is not None]
    capped_gradients = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gradients if grad is not None]
    train_optimizer = optimizer.apply_gradients(capped_gradients)
    return loss, train_optimizer


class RecognitionRNN(object):
    
    def __init__(self, batch_size, window_size, learning_rate=0.8, hidden_layer_size=512, number_of_layers=2, 
                 dropout=True, keep_prob=0.8, size=12, gradient_clip_margin=4):
    
        self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 2], name='input_data')
        self.targets = tf.placeholder(tf.float32, [batch_size, 12], name='targets')

        # cell, init_state = LSTMCell(hidden_layer_size, batch_size, number_of_layers, dropout, keep_prob)
        cell, init_state = LSTMCell(numNeuron, self.inputs, len(numNeuron), keep_prob)

        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)

        self.logits = outLayer(outputs, size)

        self.loss, self.opt = opt_loss(self.logits, self.targets, learning_rate)


if __name__ == "__main__":
    numNeuron = [128, 64]
    trainX, testX, trainY, testY = preprocess.getTrainTest("mfcc")
    batchSize, windowSize = preprocess.getBatchWindow(trainX)
    print("trainX : ", trainX.shape)
    # x_input = tf.placeholder(tf.float32, [batchSize, windowSize, 2]) 
    # cell, init_state = LSTMCell(numNeuron, x_input, len(numNeuron), keep_prob)

    # lstmOut, final_state = tf.nn.dynamic_rnn(cell, x_input, initial_state=init_state)
    # preds = outLayer(lstmOut, 10)
    # y_out = tf.placeholder(tf.float32, [batchSize, 1])
    # loss, opt = opt_loss(preds, y_out, 0.8) 

    model = RecognitionRNN(batchSize, windowSize)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        num_epochs = 10
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = len(trainX) // batchSize

            for batch in range(num_batches):
                start_idx = batch * batchSize
                end_idx = start_idx + batchSize

                batch_X = trainX[start_idx:end_idx]
                batch_Y = trainY[start_idx:end_idx]

                _, loss = sess.run([model.opt, model.loss], feed_dict={model.inputs: batch_X, model.targets: batch_Y})
                total_loss += loss

            avg_loss = total_loss / num_batches
            print("Epoch:", epoch+1, "Loss:", avg_loss)