import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import torch
from torchviz import make_dot

sampleData = np.array([
  [170, 58.8],
  [175, 61.5],
  [158, 43],
  [163, 46],
  [182, 80]
])

# 신장 데이터를 입력으로 가지고, 몸무게를 출력으로 내어 몸무게를 예측하는 모델을 만들 것
# 따라서 신장 데이터가 X, 몸무게 데이터가 Y

X = sampleData[:,0]
Y = sampleData[:,1]

# 예측 모델을 만들 때는 값이 작아야 편하므로, 값을 작게 만들어준다.

X = X - X.mean()
Y = Y - Y.mean()

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

# 이제 예측 모델에 쓰일 함수를 정의할 것.
# 기울기와 y 절편을 잡아서 얘네를 손실 경사 계산을 통해 파라미터 최적화를 할 것임.
# 파라미터를 tensor 로 정의하기, 초기 값은 1로 설정

W = torch.tensor(1.0, requires_grad=True).float()
B = torch.tensor(1.0, requires_grad=True).float()

def pred(X): # 예측 함수
  return W * X + B

X_max = X.max()
X_min = X.min()

X_range = np.array((X_min, X_max))
X_range = torch.from_numpy(X_range).float()

Yp_range = pred(X_range)

# 처음 정의 된 예측 모델을 바탕으로 prediction 값을 얻기.

# 계산 그래프 얻어내기

# g = make_dot(Yp, params={'W':W, 'B':B})

# 예측 계산 그래프 시각화하기
# display(g) 

# 손실 계산 하는 함수 정의

def mse(Yp, Y): # 손실 함수
  loss = ((Yp - Y) ** 2).mean()
  return loss

# loss = mse(Yp, Y)

# g = make_dot(loss, params={'W':W, 'B':B})

# 손실 계산 그래프 시각화하기
# display(g) 

# 경사 계산 (파라미터 별로 편미분)

num_epochs = 500
lr = 0.001
history = np.zeros((0, 2))

import torch.optim as optim
optimizer = optim.SGD([W, B], lr=lr, momentum=0.9)

for epoch in range(num_epochs):
  Yp = pred(X)
  loss = mse(Yp, Y)
  loss.backward() # 예측 함수와 손실 함수의 합성 함수인 손실

  # with torch.no_grad():
  #   W -= lr * W.grad
  #   B -= lr * B.grad

  #   W.grad.zero_()
  #   B.grad.zero_()

  optimizer.step()
  optimizer.zero_grad()

  if (epoch % 10 == 0):
    item = np.array([epoch, loss.item()])
    history = np.vstack((history, item))
    print(f'epoch = {epoch} loss = {loss:.4f}')

# 예측 모델을 최적화 결과

print('W =', W.data.numpy())
print('B =', B.data.numpy())

print(f'초기상태 : 손실 : {history[0,1]:.4f}')
print(f'최종상태 : 손실 : {history[-1,1]:.4f}')

# 손실이 어떻게, 얼마나 줄었는지 확인하기

plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('반복 횟수')
plt.ylabel('손실')
plt.title('학습 곡선')
plt.show()

# 내가 최적화 한 파라미터를 가진 선형 예측 함수 확인해보기

X_max = X.max()
X_min = X.min()

X_range = np.array((X_min, X_max))
X_range = torch.from_numpy(X_range).float()

Y_range = pred(X_range)


plt.scatter(X, Y, c='k', s=50)
plt.plot(X_range.data, Y_range.data, lw=2, c='b')
plt.show()

plt.scatter(X, Y, c='k', s=50)
plt.plot(X_range.data, Yp_range.data, lw=2, c='b')
plt.show()





