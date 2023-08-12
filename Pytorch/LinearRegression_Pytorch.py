import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

num_epochs = 10

print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu')

# (근무일 수, 얻은 총 피로량) 
# (100, 105), (125, 122), (150, 155), (190, 176), (206, 207)

x = [100, 125, 150, 190, 206]
y = [105, 122, 155, 176, 207]

# 주어진 데이터와 최대로 비슷한 상관 관계를 가장 잘 나타내는 y=Wx를 선형회귀로 추론하는 과정

class Linear_Model(nn.Module):
    def __init__(self):
        super(Linear_Model, self).__init__()

        self.layer1 = nn.Linear(1,1, bias=False) # bias 없이 y=Wx를 확인

    def forward(self, x):
        out = self.layer1(x)
        return out


# Loss Graph를 위해 epoch마다 cost를 append할 빈 리스트
Loss_list = []

# 모델 정의 및 gpu에 할당
model = Linear_Model().to(device)

# Optimizer 정의
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# 데이터 Tensor화 및 gpu에 할당
train_x = torch.Tensor(x).to(device)
train_y = torch.Tensor(y).to(device)

train_x = torch.reshape(train_x, (len(train_x), 1))
train_y = torch.reshape(train_y, (len(train_y), 1))

# tqdm으로 progress 확인을 위해 tqdm 객체 생성
num_e = tqdm(range(num_epochs))

for epoch in num_e:
    out = model(train_x)

    cost = F.mse_loss(train_y, out).to(device)

    Loss_list.append(cost.item())

    # 학습
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 2 == 0:
    # 5번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, num_epochs, cost.item()
      ))

# Weight 확인
for weight in model.parameters():
    print(weight)

# tqdm 객체 닫음
num_e.close()

# loss graph 출력

plt.plot(Loss_list[1:])
plt.show()

# 테스트 데이터 넣어보기
model.eval()
a = np.array([[175]])
print("If", a[0, 0], "is the input, the output is:", model(torch.Tensor(a).to(device)).item())
