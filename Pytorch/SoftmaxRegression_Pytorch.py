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

# data-04-zoo.csv 데이터셋을 이용할 것이다.
zoo = np.loadtxt('data-04-zoo.csv', delimiter=',')

print(zoo.shape) # (101, 17), 마지막 column은 0~6 중 어느 분류인지 표시 되어있다

# Softmax 모델을 만들어보자
class Softmax_Model(nn.Module):

    def __init__(self):
        super(Softmax_Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(16, 7),
            nn.Softmax()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        return out


# 데이터의 label을 one-hot vector로 만들자
x = zoo[:-10, :-1]
y = zoo[:-10, -1:]

print(x.shape, y.shape) # (101, 16) (101, 1)

one_hot_y = np.zeros((len(y), int(y.max()+1)))
for num in range(len(y)):
    one_hot_y[num, int(y[num, 0])] = 1

# 이제부터 이 one hot y는 우리의 y가 됩니다.
y = one_hot_y

# 이 데이터들을 이제 torchtensor로 변환할 것입니다.
X_train = torch.Tensor(x).to(device)
Y_train = torch.Tensor(y).to(device)

# 모델 정의 후 gpu에 할당
model = Softmax_Model().to(device)

# Loss 함수 정의 및 Optimzer 정의
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(1000):
    
    X_out = model(X_train)
    cost = criterion(X_out, Y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if i % 100 == 99:
        print("Epoch:", i+1, '{:.4f}'.format(cost.item()))


x_test = zoo[-10:, :-1]
y_test = zoo[-10:, -1:]


X_test_ = torch.Tensor(x_test).to(device)
Y_test_ = torch.Tensor(y_test).to(device)

X_test_result = np.argmax(model(X_test_).detach().cpu().numpy(), axis=1)
Y_test_result = Y_test_.detach().cpu().numpy()

print(X_test_result)
print(Y_test_result)

count = 0
for i in range(len(X_test_result)):
    if X_test_result[i] == Y_test_result[i]:
        count += 1

print("Acc. : {:.2f}%".format(count/len(X_test_result)))
