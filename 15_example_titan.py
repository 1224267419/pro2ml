import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
#调用决策树类,使决策树可视化

def demo1():
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


def demo2():
    df_train = pd.read_csv("./titan_dataset/train.csv")
    df_test = pd.read_csv("./titan_dataset/test.csv")

    x_train = df_train[["Pclass", "Age", "Sex"]]
    y_train = df_train["Survived"]
    x_test = df_test[["Pclass", "Age", "Sex"]]


    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

    # 实例化字典转向量类(one-hot
    transfer = DictVectorizer(sparse=False)

    # 对于x转换成字典数据x.to_dict(orient="records")
    x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
    x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

    # 4.机器学习(决策树)
    estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)  # 实例化一个决策树对象，可以进行参数调优
    estimator.fit(x_train, y_train)  # 训练模型

    # 5.模型评估
    y_pred = estimator.predict(x_test)
    print(y_pred)
    y_pred=pd.DataFrame(y_pred)
    # y_pred.to_csv("./titan_dataset/y_pred.csv", index=False)
    export_graphviz(estimator, out_file="./titan_dataset/tree.dot",
                    feature_names=['age', 'pclass','女性', '男性'])
#根据模型进行保存决策树,并保存为dot格式
#生成的树可以在http://webgraphviz.com/中查看

if __name__ == '__main__':
    demo1()
    # demo2()