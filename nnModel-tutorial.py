import os
import torch
from torch import nn # 신경망 구성하는 데에 필요한 모든 구성 요소! 이 패키지로 신경망 만듦.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import torch.nn as nn
# import torch.nn.functional as F

# class MyModel(nn.Module): # nn Module 을 상속 받아서 기본적인 기능들을 사용할 수 있게 
#     def __init__(self): # 신경망 레이어의 구성 요소 정의
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5)
#         self.conv2 = nn.Conv2d(20, 20, 5)

#     def forward(self, x): # 호출되면 수행하는 연산
#         x = F.relu(self.conv1(x))
#         return F.relu(self.conv2(x))
    
# 하나의 클래스 안에 init 과 forward 문은 모두 반드시 정의되어야 한다.

# FasionMNIST 신경망을 만들 때 사용되는 Layer 들
# 1. nn.Flatten : 계층을 초기화하여 각 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환합니다. 
# 차원을 줄이는 거지. Flatten 이니까. 입력 텐서를 평탄화 하는 작업 n 차원을 1 차원으로 !!






device = ( # cuda 나 mps 가 사용 가능하면 사용하고, 아니면 cpu 로 돌린다.
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module): # nn Module 을 상속 받아서 기본적인 기능들을 사용할 수 있게 
    def __init__(self): # 신경망 레이어의 구성 요소 정의
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential( # Sequential 은 순서를 가지는 모듈의 컨테이너
            nn.Linear(28*28, 512),
            # input 텐서 크기가 28*28, output 텐서 크기는 512가 되도록 선형 변환을 수행 https://wikidocs.net/194943
            # 이 클래스는 가중치와 편향을 학습해서 입력 텐서에 가중치 행렬을 행렬 곱한 뒤, 편향을 덧셈한다.
            nn.ReLU(), 
            # ReLU 클래스는 입력 텐서의 각 요소에 대해 함수를 수행하는데
            # ReLU(x) = max(0, x) x 는 입력 텐서의 값이고 양수면 x, 음수면 0 을 반환
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x): # 호출되면 수행하는 연산
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)