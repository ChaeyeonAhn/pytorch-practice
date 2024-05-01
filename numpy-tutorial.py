import torch
import numpy as np
import numpy.random as npr

a = np.arange(0, 15, 1).reshape(3,5) # np.arange(15)
# ndarray 를 생성
# np.arange = 0 이상 15 미만 1 씩 차이 나게 생성

x = np.linspace(0, 99, 100) # 0 부터 99 까지 100 등분! * 끝도 포함


print(a)

print(a.size) # 행렬의 크기 (요소의 개수)

print(a.ndim) # 행렬의 차원

print(a.shape) # 행렬의 모양 (행 ? 개, 열 ? 개)

print(a.dtype) # 행렬에 들어 있는 요소의 타입

print(a.itemsize) # 행렬에 들어 있는 요소의 크기 (단위 : byte)

b = np.array([1, 2, 3]) # 안에 List 를 넣어서 ndarray 생성 가능
# print(b)

print(np.zeros((4,4))) # 4*4 짜리 0 행렬

print(np.ones((3,3))) # 3*3 짜리 1 행렬

print(np.empty((2,3))) # 초기화 되지 않은 임의의 값이 들어있는 행렬


data = [[1, 2],[3, 4]] # array
x_data = torch.tensor(data) # tensor

np_array = np.array(data) # ndarray
x_np = torch.from_numpy(np_array) # ndarray to tensor

x_ones = torch.ones_like(x_data) # x_data의 shape, type 을 유지하면서 1 행렬을 생성
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 type 을 덮어쓴다. dtype 을 따로 정의하기 때문에
print(f"Random Tensor: \n {x_rand} \n")

# 무작위(random) 또는 상수(constant) 값을 사용하기: 부터 !








