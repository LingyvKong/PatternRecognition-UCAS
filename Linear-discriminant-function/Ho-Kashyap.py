# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 02:43:47 2020

@author: 92031
"""

import numpy as np
import matplotlib.pyplot as plt

file = "E:\\UserData\\Agkd\\Course\\PatternRecognition\\data.txt"
SampleNum = 10
MaxIter = 100000

class HoKashyap():
    def __init__(self, yita = 0.01, b0=0.3):
        self.a = np.array([0, 0, 0])
        self.b = np.ones((2*SampleNum,1)) * b0  # b0>0防止收敛到0
        self.yita = yita
        self.bmin = np.ones((2*SampleNum,1)) * 1e-3

    def read(self,file,c1,c2):
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip().split(' ') for line in lines]
            y2 = np.array(lines[SampleNum*(c2-1):SampleNum*c2],dtype=float)
            y1 = np.array(lines[SampleNum*(c1-1):SampleNum*c1],dtype=float)
            y1[:,2] = 1
            y2[:,2] = 1
            y2 = -y2
            self.y1 = y1
            self.y2 = y2
            self.y = np.concatenate((y1, y2),axis=0)

    def train(self):
        self.i = 0
        GoOnflag = True
        while GoOnflag and self.i < MaxIter:
            self.i = self.i+1
            e = np.matmul(self.y, self.a).reshape((-1,1)) - self.b
            eplus = 1/2 * (e + np.abs(e))
            self.b = self.b + 2*self.yita*eplus
            self.a = np.matmul(np.linalg.pinv(self.y), self.b)
            GoOnflag = ~np.all(np.abs(e)-self.bmin <= 0)
    
    # y1为正样本(type=ndarray(10,3))， name1为正样本类别标号(string)，
    # y2为负样本(type=ndarray(10,3))，name2为负样本类别标号(string)
    # a为决策面系数矩阵 type=ndarray(3,1)
    def draw(self, title, c1, c2):
        fig ,axes = plt.subplots()
        n = axes.scatter(x=self.y1[:,0], y=self.y1[:,1], s=20, c='r', marker='^')
        m = axes.scatter(x=-self.y2[:,0], y=-self.y2[:,1], s=20, c='g', marker='*')
        plt.legend((n,m), ('$\omega_'+ str(c1)+'$', '$\omega_'+str(c2)+'$'))
        xx = np.arange(-10, 10, 1)
        b = -self.a[0]/self.a[1] * xx - self.a[2]/self.a[1]
        axes.plot(xx, b)
        axes.set_title(title)
        axes.set_xlabel('x1')
        axes.set_ylabel('x2')
        
if '__name__'=='__main__':
    q21 = HoKashyap()
    q21.read(file,1,3)
    q21.train()
    q21.draw('Apply HoKashyap to $\omega_1$ and $\omega_3$', 1,3)
    print("1 and 3 iter: {}".format(q21.i))
    print("1 and 3 a: \n{}".format(q21.a))
    
    q22 = HoKashyap(yita=0.001)
    q22.read(file,2,4)
    q22.train()
    q22.draw('Apply HoKashyap to $\omega_2$ and $\omega_4$', 2,4)
    print("2 and 4 iter: {}".format(q22.i))
    print("2 and 4 a: \n{}".format(q22.a))

