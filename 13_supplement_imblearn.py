import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

def counter(y_resampled):
    # numpy计算各元素出现次数
    unique, count = np.unique(y_resampled, return_counts=True)
    data_count = dict(zip(unique, count))
    print(data_count)

def ran_over_sam(x, y):
    # 1.随机过采样
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x, y)
    # 查看结果

    # 过采样后样本结果
    # Counter({2: 4674, 1: 4674, 0: 4674})

    counter(y_resampled)
     # 数据集可视化
    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
    plt.title("ran_over_sam")
    plt.show()
    return y_resampled

def smote(x,y):
    # 2.SMOTE过采样
    X_resampled, y_resampled = SMOTE().fit_resample(x, y)

    # 采样后样本结果
    # [(0, 4674), (1, 4674), (2, 4674)]
    counter(y_resampled)

    # 数据集可视化
    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
    plt.title("smote")
    plt.show()
    return y_resampled

def ran_und_sam(X,y):
    # 随机⽋采样
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(counter(y_resampled) )
    # 采样后结果
    # [(0, 64), (1, 64), (2, 64)]

    # 数据集可视化
    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
    plt.title("ran_under_sam")
    plt.show()
if __name__=='__main__':
    # 使用make_classification生成样本数据
    x, y = make_classification(n_samples=5000,  # 样本数
                               n_features=2,  # 特征个数=n_informative
                               n_informative=2,  # 多信息特征z个数
                               n_redundant=0,  # 冗余信息，informative特征的随机线性组合
                               n_repeated=0,  # 重复信息，随机提取n_informative和n_redundant特征
                               n_classes=3,  # 分类类别
                               n_clusters_per_class=1,  # 某一个类别是由几个cluster构成的
                               weights=[0.01, 0.05, 0.94],  # 列表类型，权重比(样本不均衡
                               random_state=0)
    print(x.shape)
    print(y.shape)

    plt.figure()  # 绘制画布
    plt.scatter(x[:,0], x[:,1],c=y)  # 第一个特征为x轴,第2个特征为y轴,c为颜色序列,即为不同类别分上不同颜色
    plt.title("origin")#原始图像
    plt.show()  # 展示画布

    ran_over_sam(x, y)#随机过采样
    smote(x,y)#SMOTE
    ran_und_sam(x,y)#随机欠采样