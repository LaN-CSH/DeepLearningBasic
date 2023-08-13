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

train_X = torch.FloatTensor(train[:, 1:])
train_Y = torch.FloatTensor(train[:, :1])
test_X = torch.FloatTensor(test[:, 1:])
test_Y = torch.FloatTensor(test[:, :1])

print(train_X.shape)
print(train_Y.shape)

print(test_X.shape)
print(test_Y.shape)

# Y를 One-hot으로 바꿔준다
Y_one_hot_ = torch.squeeze(F.one_hot(train_Y.to(torch.int64), num_classes=10))
Y_one_hot_test = torch.squeeze(F.one_hot(test_Y.to(torch.int64), num_classes=10))

# MNIST 분류모델 구성
model = nn.Sequential(
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
    nn.Softmax()
)

# Loss, Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

n_ephocs = 500
batch_size = 3000

train_X = train_X/255

for i in range(n_ephocs):
  for j in range(int(len(train_X)/batch_size)):
     outputs = model(train_X[(j*batch_size):((j+1)*batch_size), :])
     loss = loss_fn(outputs, Y_one_hot_[(j*batch_size):((j+1)*batch_size), :].to(torch.float32))
     #print(outputs[:5], torch.squeeze(Y_one_hot_[(j*batch_size):((j+1)*batch_size), :]).to(torch.float32)[:5])

     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
  if (i+1)%50 == 0:
    print("epoch{0} LOSS：{1}".format(i+1, loss.item()))

model.eval()
test_X = test_X/255
predict = model(test_X)
predict = torch.argmax(predict, dim=1)
print(predict[:5].to(torch.int32))
print(torch.squeeze(test_Y[:5, :]).to(torch.int32))

correct = 0
for i in range(len(test_X)):
  if predict[i].to(torch.int32) == torch.squeeze(test_Y[:, :]).to(torch.int32)[i]:
    correct += 1

print("ACC: {0}%".format(correct/len(test_X)))