import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def demo0():#分层随即划分的用法
    X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2]])
    Y = np.array([0, 0, 0, 1, 1])
    ss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)

    for train_index, test_index in ss.split(X, Y):
        #注意,这个分类器返回的是索引而非真实值,因此下面的数字均为索引
        # print("训练集索引",train_index)
        for i in train_index:
            print("训练集特征值为",X[i],"目标值为",Y[i])
        # print("测试集索引",test_index)
        for i in train_index:
            print("测试集特征值为",X[i],"目标值为",Y[i])
def demo1():#业务代码
    data=pd.read_csv("./otto_dataset/train.csv")
    #
    # sns.countplot(data.target)
    # plt.show() #由图可知数据不均衡
    # print(data.describe())
    # print(data.shape)

    rus = RandomUnderSampler(random_state=0)

    #选择需要的特征,丢弃不要的特征
    y = data["target"]
    x = data.drop(["id", "target"], axis=1)#axis为操作的维度

    #执行随机欠采样,使得各类别的数据数量相同
    X_resampled, y_resampled = rus.fit_resample(x,y)
    # print(X_resampled.shape)
    # print(y_resampled.shape)

    #可视化
    # sns.countplot(y_resampled)
    # plt.show()


    # 将标签值转换为编码
    le = LabelEncoder()
    y_resampled = le.fit_transform(y_resampled)
    print(y_resampled)

    #拆分数据集
    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,random_state=22)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    #分层采样,nsplist即仅使用分层采样而不重复采样
    for train_index, test_index in sss.split(x_train, y_train):
        x_train=X_resampled.values[train_index]
        x_val=X_resampled.values[test_index]

        y_train=y_resampled[train_index]
        y_test=y_resampled[test_index]

    # sns.countplot(y_resampled)
    # plt.show()

    #数据标准化
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    origin_feature_num=x_val_scaled.shape[1]
    pca = PCA(n_components=0.9)#保留特征中90%的信息

    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)
    # print("原数据特征数目为:",origin_feature_num)
    # print("表达90%的信息只需要:",x_train_pca.shape[1],"个特征")
    # 原数据特征数目为: 93
    # 表达90 % 的信息只需要: 65个特征





if __name__ == '__main__':
    # demo0()
    demo1()