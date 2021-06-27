import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio, os
from matplotlib import cm
from setting import *

__author__ = 'kly'

"""
activation and loss function
"""

def sigmoid(x):
    # Our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def diff_sigmoid(x):
    z = sigmoid(x)
    return z * (1-z)

def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

def diff_tanh(x):
    z = tanh(x)
    return 1 - np.square(z)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

    
"""
神经网络
"""
class Linear():
    def __init__(self, in_features, out_features, func, lr, bias=True):
        np.random.seed(SEED)
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr
        self.activation = func
        self.bias_flag = bias
        if self.activation==sigmoid:
            self.diff_act = diff_sigmoid
        elif self.activation==tanh:
            self.diff_act = diff_tanh
        self.weight = np.random.rand(in_features, out_features)
        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = None
        self.reset_parameters()
        
    def reset_parameters(self):
        pass
        
    def forward(self, inputs):
        self.rst = np.dot(inputs, self.weight)
        if(self.bias_flag):
            self.rst += self.bias
        return self.activation(self.rst)
    
    # Delta为当前节点收集到的加权误差，yh为上一层的输出
    # 返回值为当前节点为上一层节点传递的误差
    def backward(self, Delta, yh):
        # TODO 当有bias时更新bias
        yh = yh.reshape([-1, self.weight.shape[0]]).T
        Delta = Delta.reshape([-1,self.weight.shape[1]])
        diff = self.diff_act(self.rst).reshape([-1,self.weight.shape[1]])
        delta = np.multiply(Delta, diff)
        yh = np.mean(yh, axis=1).reshape([self.weight.shape[0], -1])
        delta = np.mean(delta, axis=0).reshape([-1,self.weight.shape[1]])
        weight = self.weight
        self.weight = self.weight + self.lr * np.dot(yh, delta)
        if self.bias_flag:
            self.bias = self.bias + self.lr * delta.reshape([-1])
        return np.dot(weight, delta.T)
        
    
class NeuralNetwork:
    def __init__(self, hidden_layer_dim, lr):
        self.l1 = Linear(3, hidden_layer_dim, tanh, lr)
        self.l2 = Linear(hidden_layer_dim, 3, sigmoid, lr)
        self.epoch_loss_log = []
    
    def forward(self, inputs):
        self.yh = self.l1.forward(inputs)
        y = self.l2.forward(self.yh)
        return y
    
    def print_rst(self, inputs):
        rst = self.forward(inputs)
        rst = rst.argmax(axis=1)
        return rst
    
    def draw_log(self,title='epoch-loss'):
        log = np.array(self.epoch_loss_log)
        plt.plot(log[:,0], log[:,1])
        plt.title(title)
        plt.show()
        return log
    
    def StochasticBP_train(self, data, y_trues, shuffle=True):
        index = np.arange(0, data.shape[0])
        for epoch in range(Epochs):
            if shuffle:
                index = np.random.randint(0, data.shape[0], (1, data.shape[0]))    
            for i in index:
                x = data[i,:]
                y = y_trues[i,:]
                y_pred = self.forward(x)
                delta = self.l2.backward(y-y_pred, self.yh)
                _ = self.l1.backward(delta, x)
                
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.forward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                self.epoch_loss_log.append([epoch, loss])
                print("Epoch %d loss: %.3f" % (epoch, loss))
                
    def BatchBP_train(self, data, y_trues, shuffle=True):
        batch_ids = np.arange(0,data.shape[0],Batch_size)
        shuffle_X = data
        shuffle_y = y_trues
        for epoch in range(Epochs):
            if shuffle:
                per = np.random.permutation(data.shape[0])		#打乱后的行号
                shuffle_X = data[per, :]		#获取打乱后的训练数据
                shuffle_y = y_trues[per, :]
            for bid in batch_ids:
                x = shuffle_X[bid:min(bid+Batch_size,data.shape[0]),:]
                y = shuffle_y[bid:min(bid+Batch_size,data.shape[0]),:]
                y_pred = self.forward(x)
                delta = self.l2.backward(y-y_pred, self.yh)
                _ = self.l1.backward(delta, x)
            
            if epoch % 4 == 0:
                y_preds = np.apply_along_axis(self.forward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                self.epoch_loss_log.append([epoch, loss])
                print("Epoch %d loss: %.3f" % (epoch, loss))
        
"""
作业函数
"""
def plot_answer(array_list, legend, title=''):
    fig = plt.figure()
    for a in array_list:
        plt.plot(a[:,0], a[:,1])
    plt.title(title)
    plt.legend(legend)
    plt.show()

        
if __name__ == "__main__":
    # load data
    data = []
    with open(data_dir, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data.append([line.strip().split(" ") for line in lines])
        
    data = np.array(data, dtype=float).reshape([30,4])
    x = data[:, :3]
    label = np.array(data[:, -1], dtype=int)
    y = np.zeros([label.shape[0],3])
    for i in range(label.shape[0]):
        y[i,label[i]] = 1
        
    # 数据集可视化
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x[:,0], x[:,1], x[:,2], s=10, c=label, cmap='jet')
    # # plt.show()
    # transition = lambda x,N: (1+np.sin(-0.5*np.pi+2*np.pi*x / (1.0*N)))/2.0
    # for i in range (40):
    # 	horiAngle=45+50*transition(i,40)
    # 	vertAngle=50+43*transition(i,40)
    	
    # 	ax.view_init(vertAngle,horiAngle)
    # 	filename='animFram/'+str('%03d'%i)+'.png'
    # 	plt.savefig(filename, dpi=192)
    # images = []
    # filenames=sorted(fn for fn in os.listdir('animFram/') if fn.endswith('.png'))
    # for filename in filenames:
    #     images.append(imageio.imread('animFram/'+ filename))
    # imageio.mimsave('gif1.gif', images,duration=0.3)
    
    # **********作业题解答**********************
    
    # build model1
    model1 = NeuralNetwork(Hidden_layer_dim[0], Lr[0])
    model1.StochasticBP_train(x, y, shuffle=False)
    print(model1.print_rst(x))
    model1.draw_log(title="StochasticBP_train")
    
    # build model2
    model2 = NeuralNetwork(Hidden_layer_dim[0], Lr[0])
    model2.BatchBP_train(x, y)
    print(model2.print_rst(x))
    model2.draw_log(title="BatchBP_train")
    
    # # Hidden_layer_dim变化对结果的影响
    hidden_log = []
    for hidden in Hidden_layer_dim:
        model = NeuralNetwork(hidden, Lr[0])
        model.StochasticBP_train(x, y, shuffle=False)
        hidden_log.append(model.draw_log(title="StochasticBP_train"))
    plot_answer(hidden_log, Hidden_layer_dim, "StochasticBP_train")
    
    hidden_log = []
    for hidden in Hidden_layer_dim:
        model = NeuralNetwork(hidden, Lr[0])
        model.BatchBP_train(x, y)
        hidden_log.append(model.draw_log(title="BatchBP_train"))
    plot_answer(hidden_log, Hidden_layer_dim, "BatchBP_train")
    
    # learning rate变化对结果的影响
    lr_log = []
    for lr in Lr:
        model = NeuralNetwork(Hidden_layer_dim[0], lr)
        model.StochasticBP_train(x, y, shuffle=False)
        lr_log.append(model.draw_log(title="StochasticBP_train"))
    plot_answer(lr_log, Lr, "StochasticBP_train")
    
    lr_log = []
    for lr in Lr:
        model = NeuralNetwork(Hidden_layer_dim[0], lr)
        model.BatchBP_train(x, y)
        lr_log.append(model.draw_log(title="BatchBP_train"))
    plot_answer(lr_log, Lr, "BatchBP_train")
    
    