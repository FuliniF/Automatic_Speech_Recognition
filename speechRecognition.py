import dvc.api
from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

with dvc.api.open("Number",
                  "https://dagshub.com/kingabzpro/Speech_Commands_Dataset/src/master") as dataset:
    