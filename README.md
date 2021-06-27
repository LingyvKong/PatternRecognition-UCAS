本项目包含以下算法的python实现：

- 线性判别函数：Linear-discriminant-function
  - 感知器（batch perception）：batch_perception.py
  - Ho-Kashyap  algorithm：Ho-Kashyap.py
  - MSE多类扩展方法：MSE.py
- 神经网络：Neural-Networks
  - 三层前向神经网络：bpnn.py
    - 隐含层结点的激励函数：双曲正切函数
    - 输出层的激励函数： sigmoid 函数
    - 目标函数：平方误差准则函数
    - 支持批量方式更新权重和单样本方式更新权重
- 聚类：Clustering
  - k-means聚类：kmeans.py
- SVM
  - 从MNIST数据集中任意选择两类，使用LIBSVM对其进行SVM分类



每个算法的结果见对应文件夹的README，有问题欢迎提 issue

