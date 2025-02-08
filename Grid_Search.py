from sklearn.datasets import load_iris  # 包内内置了部分数据集
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # 标准化

iris = load_iris()  # 加载数据集,
transfer = StandardScaler()  # 实例化转换器类

x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=22,
                                                    stratify=iris['target'])  # 划分数据集

# 数据标准化
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

estimator = KNeighborsClassifier()
param_dict = {"n_neighbors": [1, 3, 5,7]}  # 字典传入要验证的参数
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5)
# cv为交叉验证的折数
#
estimator.fit(x_train, y_train)  # 训练模型
# print('预测值是', result == y_test)  # 比较预测结果和测试集结果
# sco = estimator.score(x_test, y_test)  # 内置方法帮助计算模型准确率
# print('准确率为', sco)
print("在交叉验证中验证的最好结果：\n",estimator.best_score_)
print("最好的参数模型：\n",estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)#不同参数的结果
