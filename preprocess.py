import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "./train/"
    
# import dataset
def getLabel(path = DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# .wav to fbank
def wav2fbank(filePath):
    # load data
    sampleRate, signal = wavfile.read(filePath)
    signal = signal[0 : int(3.5 * sampleRate)]

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

# .wav to mfcc
def wav2mfcc(filePath, numCepstral = 12):
    fbank = wav2fbank(filePath)
    mfcc = dct(fbank, type = 2, axis = 1, norm = "ortho")[ : , 1 : (numCepstral + 1)]
    cepLifter = 23
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cepLifter / 2) * np.sin(np.pi * n / cepLifter)
    mfcc *= lift

    return mfcc

def save_data(feature_type, path = DATA_PATH):
    labels, _, _ = getLabel()
    if feature_type == "mfcc":
        for label in labels:
            vectors = []
            wavfiles = [path + label + "/" + wavfile for wavfile in os.listdir(path + "/" + label)]
            for wavfile in wavfiles:
                mfcc = wav2mfcc(wavfile)
                # normalize
                mfcc = (mfcc - (np.mean(mfcc, axis = 0) + 1e-8)) / np.std(mfcc, axis = 0)
                vectors.append(mfcc)
        
            np.save(label + "_mfcc", vectors, allow_pickle=True)

    elif feature_type == "fbank":
        for label in labels:
            vectors = []
            wavfiles = [path + label + "/" + wavfile for wavfile in os.listdir(path + "/" + label)]
            for wavfile in  wavfiles:
                fbank = wav2fbank(wavfile)
                # normalize
                fbank = (fbank - (np.mean(fbank, axis = 0) + 1e-8)) / np.std(fbank, axis = 0)
                vectors.append(fbank)
                
            np.save(label + "_fbank.npy", vectors, allow_pickle=True)

    return


# def get_train_test(feature_type, splitRatio = 0.6, randomState = 42):
#     labels, indices, _ = getLabel()
#     # print(labels)

#     X = np.load(labels[0] + "_" + feature_type + ".npy", allow_pickle= True)
#     print(X)
#     y = np.zeros(X.shape[0])

#     for i, label in enumerate(labels[1 : ]):
#         x = np.load(label + "_" + feature_type + '.npy', allow_pickle=True)
#         X = np.vstack((X, x))
#         y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

#     assert X.shape[0] == len(y)

#     return train_test_split(X, y, test_size= (1 - splitRatio), random_state=randomState, shuffle=True)


# test
# feature_type = "fbank"
# save_data(feature_type)
# npyRead = np.load("down_mfcc.npy", allow_pickle=True)
# print(npyRead.shape())