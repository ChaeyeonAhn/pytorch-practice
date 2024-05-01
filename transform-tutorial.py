import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# 모든 TorchVision 데이터 셋들은 매개변수 두 개를 가짐 : 
# 특징(feature)을 변경하기 위한 transform, 정답(label)을 변경하기 위한 target_transform

# FashionMNIST 의 feature은 PIL Image 형식
# label 은 integer 형식

# 학습을 하려면 정규화(normalize)된 텐서 형태의 특징(feature)과 
# 원-핫(one-hot)으로 부호화(encode)된 텐서 형태의 정답(label)이 필요.
# 이러한 변형(transformation)을 하기 위해 ToTensor 와 Lambda 를 사용합니다.

ds = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(), # 정규화된 텐서 형태로 만들기

    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # 부호화된 텐서
    # 여기서 일단 10칸 짜리 0으로 된 텐서를 만들고
    # 뒤에 scatter 는 label y 가 주는 인덱스에 value = 1을 할당함 -> 원 핫 인코딩


)

# ToTensor() 란
# PIL Image나 NumPy ndarray 를 Float Tensor 로 변환하고,
# 이미지 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)합니다.

# One-hot 인코딩이란?
# 어떤 단어를 숫자로 표현하려면 단어마다의 숫자를 부여하는 방식으로 처리함 (마치 인덱스를 부여하듯이)
# 근데 원 핫 인코딩은 내가 표현하고 싶은 단어에만 1을 부여, 나머지에는 0을 부여하고
# 전체 단어 집합의 크기를 벡터의 차원으로 하여 하나의 벡터로 만드는 방식
