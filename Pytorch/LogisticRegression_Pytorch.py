# 토치 식구들
import torch
import torch.nn as nn
import torch.nn.functional as F

# 넘파이는 가족이다
import numpy as np

# for문 진행현황을 멋있게 보는 방법
from tqdm import tqdm

# loss 추이 등 그래프를 찍어볼 때 사용할 것이다.
import matplotlib.pyplot as plt



# Hyperparameters
# 해당 예시 코드의 경우 사용한 데이터 예시가 Sigmoid로 끼워넣기에 빡빡한 모양새를 가지고 있어서
# LR을 좀 작게 설정하고 num_epoch을 늘렸다
num_epochs = 20000
lr = 0.003

# GPU 체크
torch.cuda.is_available()


# GPU 있으면 GPU 씁시다
device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu')


# 데이터
# (종합 점수, 합격 여부) -> 0이면 불합격, 1이면 합격
# (15, 0 ), (24, 0), (57, 0), (78, 1) (90, 1), (114, 1)

x = [15, 24, 57, 78, 90, 114]
y = [0, 0, 0, 1, 1, 1]

# 데이터 Tensor화 및 gpu에 할당
train_x = torch.Tensor(x).to(device)
train_y = torch.Tensor(y).to(device)

# (n, 1) 꼴로 바꿔주자 
train_x = torch.reshape(train_x, (len(train_x), 1))
train_y = torch.reshape(train_y, (len(train_y), 1))


# 주어진 데이터와 최대로 비슷한 상관 관계를 가장 잘 나타내는 y=Wx+b에 sigmoid를 씌운 형태의 함수를 추론하는 과정


class Logistic_Model(nn.Module):
    def __init__(self):
        super(Logistic_Model, self).__init__()
        
        # Hidden Layer 없이 크기 1 Input 받아서 크기 1 Output을 출력
        self.layer1 = nn.Sequential(nn.Linear(1,1),
                                     nn.Sigmoid())
        

    def forward(self, x):
        out = self.layer1(x)
        return out


# Loss Graph를 위해 epoch마다 cost를 append할 빈 리스트
Loss_list = []


# 모델 정의 및 gpu에 할당
model = Logistic_Model().to(device)


# Loss 함수 정의
criterion = nn.BCELoss().to(device)



# Optimizer 정의
optimizer = torch.optim.SGD(model.parameters(), lr=lr)



# tqdm으로 progress 확인을 위해 tqdm 객체 생성
num_e = tqdm(range(num_epochs))


for epoch in num_e:
    out_ = model(train_x)

#     cost = criterion(train_y, out).to(device)
    cost = F.binary_cross_entropy(out_, train_y).to(device)
    
    Loss_list.append(cost.item())

    # 학습
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 1000 == 999:
    # 10번마다 로그 출력
      print(out_)
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch+1, num_epochs, cost.item()
      ))


num_e.close()


# Weight 확인
for weight in model.parameters():
    print(weight)

# loss graph 출력

plt.plot(Loss_list[1:])
plt.show()


# ## 10000번째 에폭을 보면 3번째 인덱스(out_[2])는 0.5가 넘는 것을 볼 수 있다.
# tensor([[0.1009],
#         [0.1541],
#         [0.5191],
#         [0.7700],
#         [0.8647],
#         [0.9589]], device='cuda:0', grad_fn=<SigmoidBackward0>)
# Epoch 10000/20000 Cost: 0.242402
#     
# 여기서 Loss는 거의 바로 수렴한 것을 볼 수 있었지만 
# 꼭 Sigmoid에 끼워서 맞추려면 10000보다는 많은 에폭이 필요하다는 것을 알 수 있다.
# Overfitting이 될 수 있음


# 테스트 데이터 넣어보기
model.eval()
a = np.array([[66]]) # model에 넣어줄 때는 (n, 1)꼴로 받아야하기 때문에
result = model(torch.Tensor(a).to(device)).item()
if result >= 0.5:
    result_ = 1
elif result < 0.5:
    result_ = 0
else:
    print("ERROR")
    
print("If", a[0, 0], "is the input, the output is:", result_)


# y = Wx + b에 Sigmoid를 적용한 그래프 그려보기

lista = list(range(0,120,1))
lista = np.array(lista)
lista = np.reshape(lista, (len(lista), 1))

plt.plot(lista, model(torch.Tensor(lista).to(device)).detach().cpu().numpy())
plt.plot(x, y, 'ro')
plt.plot(lista, 1/2*np.ones(len(lista)), 'g--')
plt.show()





