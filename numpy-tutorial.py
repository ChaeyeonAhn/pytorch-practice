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

x_rand = torch.rand_like(x_data, dtype = torch.float) # x_data의 type 을 덮어쓴다. dtype 을 따로 정의하기 때문에
print(f"Random Tensor: \n {x_rand} \n")

print(x_data[:, 0]) # x_data 의 0, 1 column 출력
# tensor 도 기본 행렬처럼 !!!

x_data1 = torch.cat([x_data, x_data, x_data], dim = 1) # 1차원으로 옆으로 concat
print(x_data1)

x_data2 = torch.stack((x_data, x_data, x_data), dim = 0) # stack 의 특징을 알아보자
print(x_data2)

x_data3 = torch.stack((x_data, x_data, x_data), dim = 1) 
print(x_data3)

x_data4 = torch.stack((x_data, x_data, x_data), dim = 2) 
print(x_data4)

# 아래 두 코드가 어떤 텐서의 transpose 를 주는 코드
y1 = x_data.T
print(y1)

y2 = x_data.matmul(x_data.T) # 자기 자신과 transpose 를 곱하기
print(y2)

y3 = x_data.sum() 
# 텐서 안의 모든 요소의 합을 산출해서 요소 1개짜리 텐서를 반환
# 해당 텐서는 요소가 하나이므로 item() 함수를 쓰면 python 의 숫자 값으로 변환 가능
print(y3.item())

x_data.add_(2) 
# _ underscore 가 의미하는 것은 x_data 의 모든 요소에 2를 더한 다음 x_data를 아예 바꿔버림
# _ 없으면 원래 거를 바꾸지 않고 그냥 더한 상태의 새로운 텐서를 내 놓는다
print(x_data)

# *** 만약 텐서를 넘파이로 변환한다면
numpy_x = x_data.numpy()

x_data.add_(3) # 이걸 하면 이거에 참조되어 변환된 넘파이 엑스도 같이 바뀐다

print(x_data)
print(numpy_x) # 같이 바뀐 넘파이가 출력됨

# *** 만약 넘파이를 텐서로 변환한다면
numpy_y = np.ones(5)
y_data = torch.from_numpy(numpy_y) # from numpy, torch 를 만든다.

np.add(numpy_y, 1, out = numpy_y) # numpy_y 에 1을 더해서 그 결과를 다시 out 에 저장
print(y_data) # 상응하는 텐서도 같이 바뀜











