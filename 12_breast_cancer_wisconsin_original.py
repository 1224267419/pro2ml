import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score  # 计算auc指标
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.preprocessing import StandardScaler  # 标准化

# 1.获取数据
df = pd.read_csv('./cancer-dataset/data.csv')  # 读取数据
print(df.head())
gr = df.groupby(["Class"])
# print(gr.count())
# 2.1缺失值处理
df.replace(to_replace='?', value=np.nan)  # 缺失值替换为空白
print(df.shape)
x = df.iloc[:, 1:-1]  # 去掉第一列和最后一列
y = df["Class"]
# print(x.head())
# print(y.head())
# 将数据集划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)

# 标准化数据
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 实例化模型
estimator = LogisticRegression()

estimator.fit(x_train, y_train)
print("准确率=", estimator.score(x_test, y_test))  # 准确率
y_pre = estimator.predict(x_test)
print(y_pre)

# 5.3评价指标
ret = classification_report(y_test, y_pre, labels=(2, 4), target_names=('良性', '恶性'))
# 为对应类别附上名称
print(ret)
y_test = np.where(y_test > 3, 1, 0)  # >3的值为1,否则为0(预测值只有2和4,分别对应良性和恶性
print("auc=", roc_auc_score(y_test, y_pre))
