import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df=pd.read_csv("./titan_dataset/titanic.csv")
x = df[["Pclass", "Age", "Sex"]]
y = df["Survived"]

# 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
x['Age'].fillna(x['Age'].mean(), inplace=True)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)

#实例化字典转向量类(one-hot
transfer = DictVectorizer(sparse=False)

# 对于x转换成字典数据x.to_dict(orient="records")
x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

# 4.机器学习(决策树)
estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)  # 实例化一个决策树对象，可以进行参数调优
estimator.fit(x_train, y_train)   # 训练模型

# 5.模型评估

print(estimator.predict(x_test))
print(estimator.score(x_test, y_test))
