from libsvm.python.svmutil import *
from libsvm.python.svm import *

import os
import struct
import numpy as np
from libsvm.python.commonutil import svm_read_problem
from libsvm.tools.grid import *
 
def load_data():
    images_path = "mnist/train-images.idx3-ubyte"
    labels_path = "mnist/train-labels.idx1-ubyte"
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
 
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    # images为n*m,n 是样本数(行数), m 是特征数784(列数)
    index = np.argwhere(labels<=1)
    return images[index[:,0], :], labels[index[:,0]]
 
 
def load_data_test():
    images_path = "mnist/t10k-images.idx3-ubyte"
    labels_path = "mnist/t10k-labels.idx1-ubyte"
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
 
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    index = np.argwhere(labels<=1)
    return images[index[:,0], :], labels[index[:,0]]
 
 
 
def data_to_libsvm(images, labels):
    path = "mnist/images_libsvm_form"
    try:
        file1 = open(path)  
        file1.close()
        print("libsvm格式的mnist训练集文件已生成，读取数据中。。。")
        return svm_read_problem(path)
 
    except IOError:
        print("libsvm格式的训练集文件未生成，开始生成数据。")
        with open(path, "w") as svmfile:
            for i in range(len(images)):
                svmfile.write(str(labels[i]) + " ")
                for j in range(len(images[i])):
                    if (images[i][j] != 0):
                        svmfile.write(
                            str(j + 1) + ":" + "%.7f" % (images[i][j] / 255) +
                            " ")
                svmfile.write("\n")
        return svm_read_problem(path)
 
 
def testdata_to_libsvm(images, labels):
    path = "mnist/test_images_libsvm_form"
    try:
        file1 = open(path)
        file1.close()
        print("libsvm格式的mnist测试集文件已生成，读取数据中。。。")
        return svm_read_problem(path)
 
    except IOError:
        print("libsvm格式的测试集文件未生成，开始生成数据。")
        with open(path, "w") as svmfile:
            for i in range(len(images)):
                svmfile.write(str(labels[i]) + " ")
                for j in range(len(images[i])):
                    if (images[i][j] != 0):
                        svmfile.write(
                            str(j + 1) + ":" + "%.7f" % (images[i][j] / 255) +
                            " ")
                svmfile.write("\n")
        return svm_read_problem(path)
    
class Svm:
    def __init__(self, svm_label, svm_images, svm_test_label, svm_test_images):
        self.svm_label = svm_label
        self.svm_images = svm_images
        self.svm_test_label = svm_test_label
        self.svm_test_images = svm_test_images
 
    def train(self, numToClassfy, numToTrain, args):
        m = svm_train(self.svm_label[:numToTrain],
                      self.svm_images[:numToTrain], args)
        p_label, p_acc, p_val = svm_predict(
            self.svm_test_label[:numToClassfy],
            self.svm_test_images[:numToClassfy], m)
        return p_label
    
if __name__ == '__main__':     
    numToTrain = 12665  #训练数据集大小
    numToClassfy = 2115 #测试数据集大小
     
    print("开始读取MNIST数据：")
    images, label = load_data()
    test_images, test_label = load_data_test()
    #libsvm格式数据读取
    svm_label, svm_images = data_to_libsvm(images, label)
    svm_test_label, svm_test_images = testdata_to_libsvm(test_images, test_label)
     
    #线性svm模型训练
    print("SVM 1训练中...")
    svm = Svm(svm_label, svm_images, svm_test_label, svm_test_images)
    svm.train(numToClassfy, numToTrain, "-m 1000 -c 1 -t 0")
     
    #RBF核svm模型训练
    print("SVM 2训练中...")
    svm = Svm(svm_label, svm_images, svm_test_label, svm_test_images)
    svm.train(numToClassfy, numToTrain, "-m 1000 -t 2")
    
    for c in range(1,4):
        for g in range(1,5):
            g = g * 1/784
            svm = Svm(svm_label, svm_images, svm_test_label, svm_test_images)
            print("RBF kernal, c={}, g={}".format(c,g))
            svm.train(numToClassfy, numToTrain, "-m 1000 -t 2 -c "+str(c)+" -g "+str(g))
    # rate, param = find_parameters('mnist/images_libsvm_form', '-log2c -5,5,2 -log2g 3,-3,-2')