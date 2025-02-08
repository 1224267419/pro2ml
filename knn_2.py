import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris  # 包内内置了部分数据集
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # 标准化

'''
datasets.load"(
·获取小规模数据集，数据包含在datasets里
datasets.fetch data_home=None)
·获取大规模据集，需要从网络上下载，函数的第一个参数是data_home,表示数据集下载的
目录，默认是~/scikit_learn data/
'''
# 数据集获取
iris = load_iris()  # 加载数据集,
# print(iris)
# news=fetch_20newsgroups_vectorized()#大型数据集
# print("鸢尾花的特征值：\n",iris['data'])
# print("鸢尾花的目标值\n",iris["target"])
print("鸢尾花的特征名\n", iris["feature_names"])
# print("鸢尾花的目标名\n",iris["target_names"])
# print("鸢尾花的描述 \n",iris["DESCR"])

iris_d = pd.DataFrame(iris['data'],
                      columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
# 第一个参数是存放在DataFrame里的数据，参数index:行名，参数columns:列名,注意行名和列名如果存在要对应正确的行列数


# 用pandas帮助处理和展示数据
iris_d['target'] = iris['target']


def plot_iris(iris, col1, col2):
    sns.lmplot(x=col1, y=col2, data=iris, hue='target', fit_reg=False)  # 点图
    # hue为目标值,fit_reg为回归直线
    plt.xlabel(col1)  # 特征1
    plt.ylabel(col2)  # 特征2
    plt.title('鸢尾花种类分布图')
    plt.show()


transfer = StandardScaler()  # 实例化转换器类
# plot_iris(iris_d, 'sepal width (cm)', 'petal length (cm)')  # 可以任选两个特征进行查看
# iris['data']=transfer.fit_transform(iris['data'])#应当先划分数据,再做标准化
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=22,
                                                    stratify=iris['target'])  # 划分数据集

# 数据标准化
x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
x_test = transfer.transform(x_test)#因为都使用标准化方法,因此上下二式子等价(相同fit)

# 此处的stratify用于确保分割的测试集和验证集拥有相同的数据比例
# 上述代码用于划分训练集和测试集,其中测试集占20%,随机数种子为22
# print("训练集的特征值：\n", x_train)
# print("训练集的目标值：\n", x_test)
# print("测试集的特征值：\n", y_train)
# print("测试集的目标值：\n", y_test)

estimator = KNeighborsClassifier(n_neighbors=5)  # 实例化KNN分类器
estimator.fit(x_train, y_train)  # 训练模型
result = estimator.predict(x_test)  # 验证模型
print('预测值是', result == y_test)  # 比较预测结果和测试集结果是否相同
sco = estimator.score(x_test, y_test)  # 内置方法帮助计算模型准确率
print('准确率为', sco)
