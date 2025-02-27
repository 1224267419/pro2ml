import numpy as np
import lightgbm as lgb
from gmpy2 import random_state
from lightgbm import early_stopping
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from tensorflow import estimator


def demo0():    #鸢尾花案例
    iris = load_iris()
    data = iris.data
    target = iris.target
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=0)

    #创建模型,这里可以使用字典来创建,方便修改和找出参数
    #模型基本训练
    gbm = lgb.LGBMRegressor(objective='regression',  #学习目标:回归
                            learning_rate=0.05, #学习率
                            # n_estimators=20, #学习轮数
                            n_estimators=90,  # 迭代次数,触发了early_stop
                            feature_fraction_seed=10 #随机数种子
                            )
    callbacks = [lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=1)]  #使用callbacks实现早停

    gbm.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='l1',
    callbacks=callbacks    ,    )  # early_stopping_rounds：loss涨的的话，那就是最多5次，后终止
    print(gbm.score(x_test, y_test))

    #cv网格搜索调优
    #找到最优参数后直接修改原模型即可
    estimators=lgb.LGBMRegressor(num_leaves=31)
    param_grid={
        "learning_rate": [0.01, 1, 0.1],
        "n_estimators": [20,40,60,80],
    }
    gbm=GridSearchCV(estimators, param_grid,cv=5)
    gbm=gbm.fit(x_train, y_train)
    print(gbm.best_score_)#输出最好结果
    #0.9311527693475288
    print(gbm.best_params_)#最好结果的参数
#{'learning_rate': 0.1, 'n_estimators': 40}



def demo1():#lightgbm训练otto数据集

    # 生成虚拟数据集
    data=pd.read_csv("./otto_dataset/train.csv")

    y = data["target"]
    X = data.drop(["id", "target"], axis=1)  # axis为操作的维度

    #将标签编码化
    le = LabelEncoder()
    y = le.fit_transform(y)
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 LightGBM 数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test,reference=train_data)

    # 设置参数
    params = {
        'objective': 'multiclass',  # 多分类任务
        'num_class': 9,  # 类别数量,otto数据集有9类
        'metric': 'multi_logloss',  # 多分类的评估指标
        'boosting_type': 'gbdt',  # 使用 GBDT 算法
        'num_leaves': 31,  # 叶子节点数量
        'learning_rate': 0.05,  # 学习率
        'feature_fraction': 0.9,  # 特征采样比例
        'bagging_fraction': 0.8,  # 数据采样比例
        'bagging_freq': 5,  # 每 5 次迭代进行一次 bagging
        'verbose': -1  # 不输出日志
    }

    # 模型训练
    bst = lgb.train(params, train_data, valid_sets=[test_data],num_boost_round=100)

    # 模型预测
    y_pred = (bst.predict(X_test, num_iteration=bst.best_iteration))
    y_pred_class = y_pred.argmax(axis=1)  # 将概率值转换为类别标签

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))
    #在不做任何数据处理的前提下,运行速度快而且准确率和f1core均有80%
    #对比18的demo2,在相同数据量的前提下,做到了更高精准率和更快的运算时间
if __name__ == '__main__':
    # demo0()
    demo1()
