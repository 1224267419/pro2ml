import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#调用随机森林,并使用网格搜索(cv)

from xgboost import XGBClassifier

def demo1():
    df = pd.read_csv("./titan_dataset/titanic.csv")
    x = df[["Pclass", "Age", "Sex"]]
    y = df["Survived"]

    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    mean=x['Age'].mean()
    x.fillna({"Age": mean}, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)


    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

    # 实例化字典转向量类(one-hot
    transfer = DictVectorizer(sparse=False)

    # 对于x转换成字典数据x.to_dict(orient="records")
    x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
    x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

    # 4.机器学习(决策树)
    estimator = RandomForestClassifier(n_estimators=20,max_depth=5)  # 实例化随机森林
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    #用字典提供参数列表,方便修改
    estimator=GridSearchCV(estimator,param_grid=param,cv=3)

    estimator.fit(x_train, y_train)  # 训练模型

    # 5.模型评估
    y_pred = estimator.predict(x_test)
    print("demo1随机森林预测的准确率为：", estimator.score(x_test, y_test))
    #demo1随机森林预测的准确率为： 0.7757847533632287

def demo2():
    df = pd.read_csv("./titan_dataset/titanic.csv")
    x = df[["Pclass", "Age", "Sex"]]
    y = df["Survived"]

    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    mean=x['Age'].mean()
    x.fillna({"Age": mean}, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)


    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

    # 实例化字典转向量类(one-hot
    transfer = DictVectorizer(sparse=False)

    # 对于x转换成字典数据x.to_dict(orient="records")
    x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
    x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

    # 4.机器学习(决策树)
    estimator = GradientBoostingClassifier(n_estimators=20,max_depth=50)  # 实例化boosting决策树
    #用字典提供参数列表,方便修改

    estimator.fit(x_train, y_train)  # 训练模型

    # 5.模型评估
    y_pred = estimator.predict(x_test)
    print("demo2的随机森林预测的准确率为：", estimator.score(x_test, y_test))
    # demo2的随机森林预测的准确率为： 0.7802690582959642
    #速度远快于使用cv搜索的随机森林,精度略低
def demo3():
    df = pd.read_csv("./titan_dataset/titanic.csv")
    x = df[["Pclass", "Age", "Sex"]]
    y = df["Survived"]

    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    mean=x['Age'].mean()
    x.fillna({"Age": mean}, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)


    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

    # 实例化字典转向量类(one-hot
    transfer = DictVectorizer(sparse=False)

    # 对于x转换成字典数据x.to_dict(orient="records")
    x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
    x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

    xg = XGBClassifier()#实例化xgboost树

    xg.fit(x_train, y_train)
    xg.score(x_test, y_test)
    print("xgboost的预测的准确率为：", xg.score(x_test, y_test))
    #随机森林预测的准确率为： 0.7802690582959642
    #默认参数下xgb的准确率为 :0.7892376681614349

def demo4():
    df = pd.read_csv("./titan_dataset/titanic.csv")
    x = df[["Pclass", "Age", "Sex"]]
    y = df["Survived"]

    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    mean=x['Age'].mean()
    x.fillna({"Age": mean}, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22)


    # 缺失值需要处理，将特征当中有类别的这些特征进行字典特征抽取
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

    # 实例化字典转向量类(one-hot
    transfer = DictVectorizer(sparse=False)

    # 对于x转换成字典数据x.to_dict(orient="records")
    x_train = transfer.fit_transform(x_train.to_dict(orient="records"))
    x_test = transfer.fit_transform(x_test.to_dict(orient="records"))

    xg = XGBClassifier()#实例化xgboost树
    depth_range = range(10)
    score = []
    for i in depth_range:
        xg = XGBClassifier(eta=1, gamma=0, max_depth=i)
        xg.fit(x_train, y_train)
        s = xg.score(x_test, y_test)
        # print(s)
        score.append(s)
    score=np.array(score)
    print("xgboost的预测的最高准确率为：", max(score),"在最大深度为",np.argmax(score),"下得到")
    import matplotlib.pyplot as plt
    plt.plot(depth_range, score)
    plt.show()
    # xgboost的预测的最高准确率为： 0.8071748878923767在最大深度为 3下得到
    #默认参数下xgb的准确率为 :0.7892376681614349  

if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
    demo4()