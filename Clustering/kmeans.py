# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:10:44 2021

@author: 92031
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

# 超参数
NUM = 5
Mu = np.array([[-2,-6],[-1,-5],[0,-4],[1,-3],[2,-2]], dtype='float')
label_mu = np.array([[1,-1],[5.5,-4.5],[1,4],[6,4.5],[9,0]], dtype='float')

# 变量初始化
epoch = 0
lines = []
flag = True
with open("X.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.strip().split(" ") for line in lines]
    
data = np.array(lines, dtype="float")
label = np.zeros([1,data.shape[0]])
for i in range(0,NUM):
    label[i*200:(i+1)*200]=i
dist = np.zeros([data.shape[0], NUM])

# 开始迭代
while flag:
    for i in range(0,NUM):
        dist[:,i] = np.sum((data-Mu[i])**2, axis=1)
    point_class = np.argmin(dist, axis=1)
    t = np.array(Mu)
    for i in range(0,NUM):
        temp = data[point_class==i,:]
        if temp.size>0:
            Mu[i] = np.mean(temp, axis=0)
    if (Mu == t).all():
        flag = False
    epoch = epoch + 1
        
# 输出迭代训练结果
print("iter num: {}".format(epoch))
plt.scatter(data[:,0], data[:,1], s=5, c=point_class)
plt.plot(Mu[:,0], Mu[:,1], 'rv')
plt.show()

# 测试精度
# 计算类别对应矩阵
class_matrix = np.zeros([NUM, NUM])  # 行坐标为聚类结果，列坐标为标签
for i in range(0,NUM):
    for j in range(0,NUM):
        class_matrix[i,j] = np.sum(point_class[i*200:(i+1)*200]==j)
class_matrix = -1 * class_matrix
row_ind,col_ind=linear_sum_assignment(class_matrix)
print(col_ind)#对应行索引的最优指派的列索引

acc = -class_matrix[row_ind,col_ind].sum()/data.shape[0]
print("class center: \n{}".format(Mu))
print("误差: \n{}".format(Mu[col_ind,:]-label_mu))
print("acc: {}".format(acc))

