import torch # 파이토치 불러오기
from torch import nn 
from torch.utils.data import DataLoader # 샘플 -> iterable 
from torchvision import datasets # 샘플
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 학습 데이터 내려 받기 / train = True
# FashionMNIST = pre-loaded dataset
training_data = datasets.FashionMNIST( 
    root="data", # 학습 데이터가 저장되는 경로
    train=True, # 학습 용이냐, 테스트 용이냐
    download=True, # root 에 데이터가 없는 경우 인터넷에서 다운로드 함.
    transform=ToTensor(), # transform to tensor
)

test_data = datasets.FashionMNIST( # 테스트 데이터 내려 받기
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# 데이터가 객체로. 1 batch = label, feature : 64 batch 
# batch_size = 64 

# # 데이터로더를 생성합니다.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8)) # 새로운 figure 생성, 창의 크기 inch 단위 설정
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() 
    # 0 ~ len(training_data) 미만의 값 tensor return. 
    # .item() : tensor to number, size = (1,) 일 때만 가능.
    # 이렇게 랜덤 인덱스를 생성

    img, label = training_data[sample_idx] # 인덱스로 데이터 셋에 접근
    figure.add_subplot(rows, cols, i) # rows cols 3*3 9 개 중에, i 번에.
    # 아래 세 줄은 이 subplot 에 대해서.
    plt.title(labels_map[label]) # title 하나 골라 설정
    plt.axis("off") # 축 안 나타냄
    plt.imshow(img.squeeze(), cmap="gray") # cmap = "gray" : 회색 이미지로 출력
plt.show()

