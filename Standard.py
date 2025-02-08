import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris  # 包内内置了部分数据集
from sklearn.preprocessing import StandardScaler,MinMaxScaler#标准化,归一化



iris = load_iris()  # 加载数据集
def maxmin_demo(iris):#归一化
    data=iris['data']
    print(data)
    transfer=MinMaxScaler(feature_range=(0,1))#实例化转换器类,以上下限均匀放缩至(0,1)区间内
    data=transfer.fit_transform(data)
    print(data)
    print(data.mean())#均值为0.5
    print(data.var())#方差


def stand_demo(iris):
    data=iris['data']
    print(data)
    transfer=StandardScaler()#实例化转换器类
    data=transfer.fit_transform(data)
    print(data)
    print(data.mean())#均值为0
    print(data.var())#方差#方差为1
maxmin_demo(iris)
# stand_demo(iris)