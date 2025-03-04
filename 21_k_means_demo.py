import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score,silhouette_score # 轮廓系数,越大越好
from sqlalchemy.util import namedtuple

def demo2(X,y):
    # 模型构建KMeans(n_clusters=2, random_state=9)
    # 模型训练+预测estimator.fit_predict(X)

    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    # 分别尝试n_cluses=2\3\4,然后查看聚类效果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()

    # 用Calinski-Harabasz Index评估的聚类分数
    print("n_clusters=2时ch分数为:",calinski_harabasz_score(X, y_pred))


def demo3(X, y):
    y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
    # 分别尝试n_cluses=2\3\4,然后查看聚类效果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()

    # 用Calinski-Harabasz Index评估的聚类分数
    print("n_clusters=3时ch分数为:",calinski_harabasz_score(X, y_pred))

def demo4(X, y):
    y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
    # 分别尝试n_cluses=2\3\4,然后查看聚类效果
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()

    # 用Calinski-Harabasz Index评估的聚类分数
    print("n_clusters=4时ch分数为:",calinski_harabasz_score(X, y_pred))


if __name__ == '__main__':
    # 1. 创建数据集

    # X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，共4个簇，
    # 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2, 0.2]
    X, y = make_blobs(n_samples=100, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2],
                      random_state=9)

    # 数据集可视化
    plt.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()

    # 2. KMeans聚类, 并使用ch方法评估
    demo2(X,y) #2个类
    demo3(X,y) #2个类
    demo4(X,y) #2个类
    '''
n_clusters=2时ch分数为: 3116.1706763322227
n_clusters=3时ch分数为: 2931.5437780930633
n_clusters=4时ch分数为: 5924.050613480169
'''
