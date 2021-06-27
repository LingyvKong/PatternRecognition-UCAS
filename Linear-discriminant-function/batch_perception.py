# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:30:44 2020

@author: 92031
"""

import numpy as np
import matplotlib.pyplot as plt
file = "E:\\UserData\\Agkd\\Course\\PatternRecognition\\data.txt"
MaxIter = 1000
a = np.array([0, 0, 0])
yita = 1

# 1.1
with open(file) as f:
    lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    y1 = np.array(lines[:10],dtype=float)
    y2 = np.array(lines[10:20],dtype=float)
    y2[:,2] = 1
    y2 = -y2

id1 = np.matmul(a, y1.T)
id2 = np.matmul(a, y2.T)

i = 0
while ~(np.all(id1>0) and np.all(id2>0)) and i<MaxIter:
    i = i+1
    a = a + yita * (np.sum(y1[id1<=0],axis=0) + np.sum(y2[id2<=0],axis=0))
    id1 = np.matmul(a, y1.T)
    id2 = np.matmul(a, y2.T)
    
print(a)
fig ,axes = plt.subplots()
n = axes.scatter(x=y1[:,0], y=y1[:,1], s=20, c='r', marker='^')
m = axes.scatter(x=-y2[:,0], y=-y2[:,1], s=20, c='g', marker='*')
plt.legend((n,m), ('$\omega_1$', '$\omega_2$'))
xx = np.arange(-10, 10, 1)
b = -a[0]/a[1] * xx - a[2]/a[1]
axes.plot(xx, b)
axes.set_title('Batch perception in $\omega_1$ and $\omega_2$')
axes.set_xlabel('x1')
axes.set_ylabel('x2')

# # 1.2
# with open(file) as f:
#     lines = f.readlines()
#     lines = [line.strip().split(' ') for line in lines]
#     y3 = np.array(lines[20:30],dtype=float)
#     y2 = np.array(lines[10:20],dtype=float)
#     y2[:,2] = 1
#     y3[:,2] = 1
#     y2 = -y2

# id3 = np.matmul(a, y3.T)
# id2 = np.matmul(a, y2.T)

# i = 0
# while ~(np.all(id3>0) and np.all(id2>0)) and i<MaxIter:
#     i = i+1
#     a = a + yita * (np.sum(y3[id3<=0],axis=0) + np.sum(y2[id2<=0],axis=0))
#     id3 = np.matmul(a, y3.T)
#     id2 = np.matmul(a, y2.T)
    
# print(a)
# fig ,axes = plt.subplots()
# n = axes.scatter(x=y3[:,0], y=y3[:,1], s=20, c='r', marker='^')
# m = axes.scatter(x=-y2[:,0], y=-y2[:,1], s=20, c='g', marker='*')
# plt.legend((n,m), ('$\omega_3$', '$\omega_2$'))
# xx = np.arange(-10, 10, 1)
# b = -a[0]/a[1] * xx - a[2]/a[1]
# axes.plot(xx, b)
# axes.set_title('Batch perception in $\omega_2$ and $\omega_3$')
# axes.set_xlabel('x1')
# axes.set_ylabel('x2')

