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

# xor 데이터
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

# Hidden Layer를 1개 넣어서 XOR 문제를 해결해보자

class XOR_model(nn.Module):
    def __init__(self):
        super(XOR_model, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(2, 4, bias=True),
            nn.Sigmoid(),
            nn.Linear(4, 4, bias=True),
            nn.Sigmoid(),
            nn.Linear(4, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.layer(x)
        return out
    
model = XOR_model().to(device)

X_data = torch.Tensor(X).to(device)
Y_data = torch.Tensor(Y).to(device)

criterion = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1)

epochs = 10000
for epoch in range(epochs):
    X_out = model(X_data)

    cost = criterion(X_out, Y_data)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 500 == 499:
        print("Epoch:", epoch+1, '{:.4f}'.format(cost.item()))

print(model(X_data).detach().cpu().numpy())