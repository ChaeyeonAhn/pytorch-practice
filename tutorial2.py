import os
import torch
import pandas as pd
from torchvision.io import read_image
import torch # 파이토치 불러오기
from torch import nn 
from torch.utils.data import DataLoader # 샘플 -> iterable 
from torchvision import datasets # 샘플
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class CustomImageDataset(torch.utils.data.Dataset):
    # 데이터 셋을 가지고 customize.
    # 아래 세 가지 함수가 필수이다.
    # 데이터 셋의 전처리, initialize 할 때 한 번만 실행됨.
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        # names = 변수명
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    # 데이터 셋의 길이 (총 샘플 수)
    def __len__(self):
        return len(self.img_labels) # 레이블의 길이 = 샘플의 수 

    # 데이터 셋에서 1 개의 샘플을 가져오는 함수
    def __getitem__(self, idx): # idx = index 
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # names 에 0 이니까, 'file_name'
        # img 경로를 생성
        image = read_image(img_path)
        # 그 경로로 이미지를 읽음 : tensor 형태로 만듦
        label = self.img_labels.iloc[idx, 1] # names 에 1 이니까, 'label'
        # 요청된 transform 을 수행.
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample
    

from torch.utils.data import DataLoader

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

# batch 란? 대량의 데이터를 일괄 처리할 때 쓴다. Batch 하나에 feature, label 하나씩, 그런 묶음이 64 개.
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 데이터 로더를 가지고 데이터 셋을 순회
train_features, train_labels = next(iter(train_dataloader)) # batch = 64 한 개씩 순회, batch 반환
# iteration 에서의 다음 item : next(iter())

print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze() # 인덱스로 접근하여 -> squeeze : 차원이 1인 것 삭제
label = train_labels[0]
plt.imshow(img, cmap="gray") 
plt.axis("off")
plt.show()
print(f"Label: {label}")