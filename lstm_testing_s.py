import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm


from model import fcmodel1
import numpy as np
import preprocess


class SpeechDataset(Dataset):
    def __init__(self, inputdata, label):
        self.inputdata = inputdata
        self.label = label

    def __len__(self):
        return len(self.inputdata)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.inputdata[idx])
        label = torch.LongTensor(self.label[idx])

        return features, label  # board_evaluation


def get_loss(label, output):
    loss = F.cross_entropy(output, label.view(-1))
    return loss
 
def get_acc(label, output):
    pred = torch.argmax(output, dim=1)
    label = label.view(-1)
    acc_sum = (label==pred).sum().item()

    return acc_sum

if __name__ == "__main__":
    print("--------------------------------------------------------")
    # preprocess.save_data()
    featureType = "mfcc"
    trainX, testX, trainY, testY = preprocess.getTrainTest(featureType)
    train_dataset = SpeechDataset(trainX, trainY)
    train_data_loader = DataLoader(
        train_dataset, batch_size=1024, num_workers=2, shuffle=True)
    

    test_dataset = SpeechDataset(testX, testY)
    test_data_loader = DataLoader(
        test_dataset, batch_size=1024, num_workers=2, shuffle=True)
    
    # print(torch.Tensor(trainX).shape)
    # print(torch.Tensor(trainY).shape)
    # print(torch.Tensor(testX).shape)
    # print(torch.Tensor(testY).shape)

    network = fcmodel1(99, 12, 12)
    optimizer = optim.SGD(network.parameters(),
                          lr=0.1)
    
    

    for i in range(10):
        train_total_acc = 0
        train_total_loss = 0
        test_total_acc = 0
        test_total_loss = 0
        for feature, label in tqdm(train_data_loader):
            optimizer.zero_grad()

            network.train()
            output = network(feature)

            loss = get_loss(label, output)
            accp = get_acc(label, output)
            train_total_loss+=loss.item()
            train_total_acc+=accp
            
            loss.backward()
            optimizer.step()
            
        print("train_avg_acc: ", train_total_acc/11400)
        print("train_avg_loss: ", train_total_loss/(11400/1024.0))
        
        for feature, label in tqdm(test_data_loader):
            network.eval()
            output = network(feature)

            loss = get_loss(label, output)
            accp = get_acc(label, output)
            test_total_loss+=loss.item()
            test_total_acc+=accp
        print("test_avg_acc: ", test_total_acc/600)
        print("test_avg_loss: ", test_total_loss)
        
            