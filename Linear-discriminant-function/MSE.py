# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:29:26 2020

@author: 92031
"""

import numpy as np
import matplotlib.pyplot as plt

file = "E:\\UserData\\Agkd\\Course\\PatternRecognition\\data.txt"
SampleNum = 10

class MSE():
    def __init__(self, yita = 0.01, b0=0.3):
        self.w = np.ones((3,4), dtype=float)

    def read(self,file,trainrt=0.8):
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip().split(' ') for line in lines]
            x1 = np.array(lines[0:SampleNum],dtype=float)
            x2 = np.array(lines[SampleNum:2*SampleNum],dtype=float)
            x3 = np.array(lines[2*SampleNum:3*SampleNum],dtype=float)
            x4 = np.array(lines[3*SampleNum:],dtype=float)
            idx = int(trainrt*SampleNum)
            self.x_train = np.transpose(np.concatenate((x1[:idx,:],
                                           x2[:idx,:],
                                           x3[:idx,:],
                                           x4[:idx,:]),axis=0))
            self.x_test = np.transpose(np.concatenate((x1[idx:,:],
                                           x2[idx:,:],
                                           x3[idx:,:],
                                           x4[idx:,:]),axis=0))
            self.x_train[2,:] = 1
            self.x_test[2,:] = 1
            self.y_train = np.zeros((4,4*8))
            self.y_test = np.zeros((4,4*2))
            for col in range(1,5):
                self.y_train[col-1][(col-1)*8:col*8]=1
                self.y_test[col-1][(col-1)*2:col*2]=1

    def train(self):
        self.w = np.matmul(np.linalg.inv(np.matmul(self.x_train, np.transpose(self.x_train)))
                           ,np.matmul(self.x_train, np.transpose(self.y_train)))
    
    def test(self, x, y, name):
        label = np.argmax(y, axis=0)
        ans = np.matmul(np.transpose(self.w), x)
        self.rst = np.argmax(ans, axis=0)
        acc = float(np.sum(self.rst==label))/label.shape[0]
        print("rst: \n", self.rst)
        print(name + "正确率为：{}\n".format(acc))
        
if '__name__'=='__main__':
    q3 = MSE()
    q3.read(file)
    q3.train()
    q3.test(q3.x_test, q3.y_test, '测试集')
    q3.test(q3.x_train, q3.y_train, '训练集')
    print("权值矩阵为：\n", q3.w)
    