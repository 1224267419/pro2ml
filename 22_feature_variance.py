import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def variance_demo(data):
    """
    删除低方差特征——特征选择
    :return: None
    """

    print("删除前形状：\n", data.shape)

    # 1、实例化一个转换器类
    transfer = VarianceThreshold(threshold=1)
    # 2、调用fit_transform
    data = transfer.fit_transform(data.iloc[:, 1:10])
    # print("删除低方差特征的结果：\n", data)
    print("删除后形状：\n", data.shape)

    return None
def pearson_demo():
    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]
    print("相关系数：", pearsonr(x1, x2))
def spearmanr_demo():
    from scipy.stats import spearmanr

    x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
    x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

    print(spearmanr(x1, x2))



def pca_demo():
    """
    对数据进行PCA降维
    :return: None
    """
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]

    # 1、实例化PCA, 小数——保留多少信息
    transfer = PCA(n_components=0.9)
    # 2、调用fit_transform
    data1 = transfer.fit_transform(data)

    print("保留90%的信息，降维结果为：\n", data1)

    # 1、实例化PCA, 整数——指定降维到的特征数
    transfer2 = PCA(n_components=3)
    # 2、调用fit_transform
    data2 = transfer2.fit_transform(data)
    print("降维到3维的结果：\n", data2)

    return None


if __name__ == "__main__":
    data = pd.read_csv("./cancer-dataset/data.csv")
    # variance_demo(data)
    # pearson_demo()
    # spearmanr_demo()
    pca_demo()