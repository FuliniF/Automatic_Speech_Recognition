import os
from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

# path define
DATA_PATH = "./train/"
SAVE_PATH = "./npyFile/"
    
# import dataset
def getLabel(path = DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# .wav file to fbank feature(not normalized)
def wav2fbank(filePath):
    # load data
    sampleRate, signal = wavfile.read(filePath)
    maxFrameLen = int(sampleRate)
    signal = signal[0 : int(sampleRate)]
    # print(type(signal))
    # print("original len : ", len(signal), maxFrameLen)
    # print(np.zeros(maxFrameLen - len(signal)))
    if len(signal) < maxFrameLen:
        add0 = maxFrameLen - len(signal)
        # print(add0)
        signal = np.append(signal, np.zeros(add0))
        # print(len(signal))
    # print("frame len : ", len(signal), "frame : ", signal)
    # os.system("pause")

    # pre-emphasis
    preEmphasis = 0.97
    emphasized = np.append(signal[0], signal[1 : ] - preEmphasis * signal[ : -1])

    # framing
    frameSize, frameStride = 0.025, 0.01
    frameLen, frameStep = int(round(frameSize * sampleRate)), int(round(frameStride * sampleRate))
    signalLen = len(emphasized)
    numFrame = int(np.ceil(np.abs(signalLen - frameLen) / frameStep)) + 1
    padSignalLen = (numFrame - 1) * frameStep + frameLen
    z = np.zeros(padSignalLen - signalLen)
    padSignal = np.append(emphasized, z)
    indices = np.arange(0, frameLen).reshape(1, -1) + np.arange(0, numFrame * frameStep, frameStep).reshape(-1, 1)
    frames = padSignal[indices]

    # window
    hamming = np.hamming(frameLen)
    frames *= hamming

    # FFT
    NFFT = 512
    magFrame = np.absolute(np.fft.rfft(frames, NFFT))
    powFrame = ((1.0 / NFFT) * (magFrame ** 2))

    # fbank
    lowFrequencyMel = 0
    highFrequencyMel = 2595 * np.log10(1 + (sampleRate / 2) / 700)
    numFilt = 40
    melPoint = np.linspace(lowFrequencyMel, highFrequencyMel, numFilt + 2)
    hzPoint = 700 * (10 ** (melPoint / 2595) - 1)
    fbank = np.zeros((numFilt, int(NFFT / 2 + 1)))
    bin = (hzPoint / (sampleRate / 2)) * (NFFT / 2)
    for i in range(1, numFilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    
    filterBank = np.dot(powFrame, fbank.T)
    filterBank = np.where(filterBank == 0, np.finfo(float).eps, filterBank)
    filterBank = 20 * np.log10(filterBank)

    return filterBank

# wav file to mfcc feature(not normalized)
def wav2mfcc(fbank, numCepstral = 12):
    mfcc = dct(fbank, type = 2, axis = 1, norm = "ortho")[ : , 1 : (numCepstral + 1)]
    cepLifter = 23
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cepLifter / 2) * np.sin(np.pi * n / cepLifter)
    mfcc *= lift

    return mfcc

def save_data(loadPath = DATA_PATH, savePath = SAVE_PATH, splitRatio = 0.05, seed = 2023):
    """
    Create npy files to store the features of datasets.
    Doesn't need to run this function if there is no change in dataset.

    ### input
        - loadPath : string

            path to the dictionary where stores the dataset

        - savePath : string

            path to the dictionary where stores the finished npy files

    ### output
        This function has no output
    """

    labels, _, _ = getLabel()
    print(labels)

    fbankVectors = []
    mfccVectors = []
    Y = []
    for label in labels:
        # print(label)
        if label == "zero":
            y = 0
        elif label == "one":
            y = 1
        elif label == "two":
            y = 2
        elif label == "three":
            y = 3
        elif label == "four":
            y = 4
        elif label == "five":
            y = 5
        elif label == "six":
            y = 6
        elif label == "seven":
            y = 7
        elif label == "eight":
            y = 8
        elif label == "nine":
            y = 9
        elif label == "up":
            y = 10
        elif label == "down":
            y = 11
        wavfiles = [loadPath + label + "/" + wavfile for wavfile in os.listdir(loadPath + label)]
        for wavfile in  wavfiles:
            fbank = wav2fbank(wavfile)
            mfcc = wav2mfcc(fbank)
            # normalize
            fbank = (fbank - (np.mean(fbank, axis = 0) + 1e-8)) / np.std(fbank, axis = 0)
            mfcc = (mfcc - (np.mean(mfcc, axis = 0) + 1e-8)) / np.std(mfcc, axis = 0)

            fbankVectors.append(fbank)
            mfccVectors.append(mfcc)
            Y.append(y)

    
    fbankTrainX, fbankTestX, mfccTrainX, mfccTestX, TrainY, TestY = train_test_split(np.asarray(fbankVectors), np.asarray(mfccVectors), np.asarray(Y),
                                                                                      test_size=splitRatio, random_state=seed, shuffle=True)
    
    np.save(savePath + "trainX_fbank", np.asarray(fbankTrainX), allow_pickle=True)
    np.save(savePath + "testX_fbank", np.asarray(fbankTestX), allow_pickle=True)
    np.save(savePath + "trainX_mfcc", np.asarray(mfccTrainX), allow_pickle=True)
    np.save(savePath + "testX_mfcc", np.asarray(mfccTestX), allow_pickle=True)
    np.save(savePath + "trainY", np.asarray(TrainY), allow_pickle=True)
    np.save(savePath + "testY", np.asarray(TestY), allow_pickle=True)

    return

def getFbank(dataPath):
    """
    Convert one wav file to fbank feature.

    ### input
        - dataPath : string

            ata path to the target wav file

    ### output
        - np 2d array

            a normalized fbank feature
    """

    fbank = wav2fbank(dataPath)
    return (fbank - (np.mean(fbank, axis = 0) + 1e-8)) / np.std(fbank, axis = 0)

def getMFCC(dataPath):
    """
    Convert one wav file to mfcc feature.

    ### input
        - dataPath : string

            data path to the target wav file

    ### output
        - np 2d array

            a normalized mfcc feature
    """

    fbank = wav2fbank(dataPath)
    mfcc = wav2mfcc(fbank)
    return (mfcc - (np.mean(mfcc, axis = 0) + 1e-8)) / np.std(mfcc, axis = 0)

def getTrainTest(feature_type):
    """
    Generate training and testing datasets from the features in npy files.

    ### input
        - feature_type : string

            Input "mfcc" for getting mfcc feature datasets and "fbank" for fbank feature datasets.

        - splitRatio : float in [0, 1]

            number of test data devided by number of hole data

        - seed : int

            random seed

    ### output (in order)
        - trainX : np 3d array

            a training data array containing 2d-arrayed features

        - testX : np 3d array

            a training data array containing 2d-arrayed features

        - trainY : np 1d array

            an array storing labels of trainX

        - testY : np 1d array

            an array storing labels of testX
    """

    # labels, _, _ = getLabel()
    # np.random.seed(seed)

    trainX = np.load(SAVE_PATH + "trainX_" + feature_type + '.npy', allow_pickle=True)
    testX = np.load(SAVE_PATH + "testX_" + feature_type + '.npy', allow_pickle=True)
    trainY = np.load(SAVE_PATH + "trainY.npy", allow_pickle=True)
    testY = np.load(SAVE_PATH + "testY.npy", allow_pickle=True)
    trainY= np.expand_dims(trainY, axis = 1)
    testY= np.expand_dims(testY, axis = 1)
    
    return trainX, testX, trainY, testY

def getBatchWindow(dataset):
    """
    Get the batch size and window size of a dataset for rnn input.

    ### input
        - dataset : np 3d array

            the input feature array

    ### output (in order)
        - batch size : int

            number of features in the array

        - window size : int

            maximum size of the features in the array

        - imbed size : int

            dimension of features

    """

    return len(dataset), len(dataset[0]), dataset.shape[2]

# compute features
# save_data()

# test
# feature_type = "fbank"
# save_data()
# trainX, testX, trainY, testY = getTrainTest(feature_type)
# _, window, imbed = getBatchWindow(testX)
# # print(testX[0].shape, testX[300].shape, max(testX[i].shape[0] for i in range(len(testX))) == min(testX[i].shape[0] for i in range(len(testX))))
# print(testY.shape)
# print(imbed)
# mfcc = getMFCC("./audio/test0.wav")
# print(mfcc)