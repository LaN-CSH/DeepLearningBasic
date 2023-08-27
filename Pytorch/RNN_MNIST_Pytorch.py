# 토치 식구들
import torch
import torch.nn as nn
import torch.nn.functional as F


# 넘파이는 가족이다
import numpy as np

# 각종 그래프 찍어볼 때 사용할 것이다.
import matplotlib.pyplot as plt

# GPU 쓸 수 있으면 쓰자
device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu')

# 데이터 불러오기
train = np.loadtxt("mnist_train.csv", delimiter=',', dtype=float)
test = np.loadtxt("mnist_test.csv", delimiter=',', dtype=float)

train_X = torch.FloatTensor(train[:, 1:]).to(device)
train_Y = torch.FloatTensor(train[:, :1]).to(device)
test_X = torch.FloatTensor(test[:, 1:]).to(device)
test_Y = torch.FloatTensor(test[:, :1]).to(device)

train_X = torch.reshape(train_X, (len(train_X), 28, 28))
test_X = torch.reshape(test_X, (len(test_X), 28, 28))

# hyperparameters
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10 

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  
        
        out = self.fc(out[:, -1, :])
        return out