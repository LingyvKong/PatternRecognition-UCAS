## 实验设置

从MNIST数据集中任意选择两类，对其进行SVM分类，可调用现有的SVM工具如LIBSVM，展示超参数C以及核函数参数的选择过程。

## 实验结果

运行前，请打开mnist文件夹，将所有压缩包（共四个）解压到当前文件夹
选择mnist数据集中的数字0和数字1两类，使用libsvm完成分类，代码见svm.py。libsvm的使用可参考guide.pdf

分别使用线性核函数和RBF核函数训练。对于RBF核函数，对c和g两个超参数使用网格化搜索。结果如下；最终选择RBF核，c=1, g=3/784，正确率为99.9527%

 

开始读取MNIST数据：

libsvm格式的mnist训练集文件已生成，读取数据中。。。

libsvm格式的mnist测试集文件已生成，读取数据中。。。

SVM 1训练中...

Accuracy = 99.9054% (2113/2115) (classification)

SVM 2训练中...

Accuracy = 99.9054% (2113/2115) (classification)

RBF kernal, c=1, g=0.0012755102040816326

Accuracy = 99.9054% (2113/2115) (classification)

RBF kernal, c=1, g=0.002551020408163265

Accuracy = 99.9054% (2113/2115) (classification)

RBF kernal, c=1, g=0.003826530612244898

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=1, g=0.00510204081632653

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=2, g=0.0012755102040816326

Accuracy = 99.9054% (2113/2115) (classification)

RBF kernal, c=2, g=0.002551020408163265

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=2, g=0.003826530612244898

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=2, g=0.00510204081632653

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=3, g=0.0012755102040816326

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=3, g=0.002551020408163265

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=3, g=0.003826530612244898

Accuracy = 99.9527% (2114/2115) (classification)

RBF kernal, c=3, g=0.00510204081632653

Accuracy = 99.9527% (2114/2115) (classification)
