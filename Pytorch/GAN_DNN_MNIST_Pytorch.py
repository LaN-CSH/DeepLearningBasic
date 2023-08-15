# 토치 식구들
import torch
import torch.nn as nn
import torch.nn.functional as F

# 이미지로 찍어보기
from PIL import Image

# 넘파이는 가족이다
import numpy as np

# 각종 그래프 찍어볼 때 사용할 것이다.
import matplotlib.pyplot as plt

# GPU 쓸 수 있으면 쓰자
device = torch.device('cuda' if torch.cuda.is_available()==True else 'cpu')

#데이터 부르기
train = np.loadtxt("mnist_train.csv", delimiter=',', dtype=float)
test = np.loadtxt("mnist_test.csv", delimiter=',', dtype=float)
print(train.shape)
print(test.shape)

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.fc = nn.Sequential(
        nn.Linear(1*28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

  def forward(self, x):
    output = self.fc(x)
    return output
  
# Latent vector의 크기는 100으로 설정
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.fc = nn.Sequential(
        nn.Linear(100, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 1*28*28),
        nn.Tanh(),
    )

  def forward(self, x):
    output = self.fc(x)
    return output
  
# tanh를 사용했기 때문에 -1~1사이로 Normalization
# 성능 향상을 위해 train data에 가우시안 노이즈 추가
train_real = ((torch.Tensor(train[:, 1:])-127.5)/127.5).to(device)
gaus_noise = torch.tensor(np.random.normal(0, 1, train_real.size()), dtype=torch.float)
test_real = torch.Tensor(test[:, 1:]).to(device)

model_D = Discriminator().to(device)
model_G = Generator().to(device)

loss_function_ = nn.BCELoss().to(device)

optim_D = torch.optim.Adam(model_D.parameters(), lr=0.0001, betas=(0.9, 0.999))
optim_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.9, 0.999))

noise = torch.randn(60000, 100).to(device)
print(type(noise), noise.shape, train_real.shape)

d_loss = []
g_loss = []
for epoch in range(500):
  for batch in range(60):
    fake_images_= model_G(noise[batch*1000:(batch+1)*1000, :])

    real_output_ = model_D(train_real[batch*1000:(batch+1)*1000, :])
    fake_output_ = model_D(fake_images_)

    # real_label_ = torch.ones_like(real_output_)
    # fake_label_ = torch.zeros_like(fake_output_)
    cost_D = (loss_function_(real_output_, torch.ones_like(real_output_)) + loss_function_(fake_output_, torch.zeros_like(fake_output_)))/2
    if epoch % 8 != 1 : # epoch % 8 !=
      model_D.zero_grad()
      cost_D.backward(retain_graph=True)
      optim_D.step()


    cost_G = loss_function_(fake_output_, torch.ones_like(fake_output_))
    model_G.zero_grad()
    cost_G.backward()
    optim_G.step()


  d_loss.append(cost_D.item())
  g_loss.append(cost_G.item())

  if epoch == 0 or epoch % 50 == 49:
    print("Epoch: {0}, cost D: {1}, cost G: {2}".format(epoch+1, cost_D.item(), cost_G.item()))


# 성능 체크 해보기
test_noise = torch.randn(25, 100).to(device)

model_G.eval()

test_samples = model_G(test_noise)
test_samples = (test_samples*127.5)+127.5
test_samples = torch.reshape(test_samples, (25, 1, 28, 28))
print(test_samples.shape)

test_samples = test_samples.detach().cpu().numpy()


plt.plot(d_loss, label="D")
plt.plot(g_loss, label="G")
plt.show()

# 이미지로 보기

ims = []
for k in range(25):
  ims.append(Image.fromarray(np.squeeze(test_samples[k, :, :, :])))

plt.figure(figsize=(5, 5))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.imshow(ims[i])
# plt.subplot(4,4,1)
# plt.imshow(im)
# plt.subplot(4,4,2)
# plt.imshow(im2)
plt.show()