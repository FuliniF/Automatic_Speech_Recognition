import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import dvc.api

with dvc.api.open("Number",
                  "https://dagshub.com/kingabzpro/Speech_Commands_Dataset/src/master") as dataset:
    
    # import dataset
    def getLabel(path = dataset):
        labels = os.listdir(path)
        label_indices = np.arange(0, len(labels))
        return labels, label_indices, to_categorical(label_indices)
    
    # .wav to mfcc
    def wav2mfcc(filePath, maxPadLen = 11):
        wave, sr = librosa.load(filePath, mono = True, sr = None)
        wave = wave[::3]

        return
