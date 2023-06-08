# Automatic Speech Recognition 
### 2023 Spring, Introduction to Artificial Intelligence
National Yang Ming Chiao Tung University  
Final Project, Group 26  
Professor: Yi-Ting, Chen

Dataset source : https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data

## code description
- preprocess.py : preprocess data with MFCC and fBank method  
- lstm_keras.py : 1st approach for LSTM model  
- lstm_torch.py : 2nd approach for LSTM model, with control group (convolutional network)  
- model.py : for lstm_torch.py to use, main model structure  
- plot.py : plot the accuracy and loss results and save .png  
- elevator.py : main UI interface, with audio recording function  
### not used code
- preprocess_test.py : for debugging :)  

## credit:  
陳昱喬 - code: preprocessing & UI design; presentation  
林怡秀 - code: LSTM model; presentation  
傅莉妮 - code: LSTM model; report (RNN LSTM); presentation  
許維也 - report (preprocess method); presentation  
