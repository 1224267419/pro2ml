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
import seaborn as sns


def counter(y_resampled):
    # numpy计算各元素出现次数
    unique, count = np.unique(y_resampled, return_counts=True)
    data_count = dict(zip(unique, count))
    print(data_count)

if __name__== '__main__':
    train = pd.read_csv("pubg_data/train_V2.csv") #读取训练数据
    #
    # print(df.head())
    # print(df.tail())
    # print(df.describe())#常用信息
    # print(df.info())#各列信息
    # print(np.unique(df['matchId']).shape)#比赛数(排除了重复项
    # print(np.unique(df['groupId']).shape)#队伍数(排除了重复项
    print(np.any(train.isnull()))#寻找空白值
    # print(df[df['winPlacePerc'].isnull()])#寻找缺失行
    train.drop(2744604, inplace=True)#只有一条缺失值,直接删除
    print(train.shape())
    print(train['playersJoined'].sort_values().head())#按升序排序,发现有的比赛只有个位数的人参加)

    # 显示每场比赛参加人数
    plt.figure(figsize=(20, 10))
    sns.countplot(train['playersJoined'])
    plt.title('playersJoined')
    plt.grid()
    plt.show()

    # 限制每局开始人数大于等于75，再进行绘制。
    # 再次绘制每局参加人数的直方图
    plt.figure(figsize=(20, 10))
    sns.countplot(train[train['playersJoined'] >= 75]['playersJoined'])
    plt.title('playersJoined')
    plt.grid()
    plt.show()
    # 75人以上参加比赛的数量比较可观,可以进行训练

    y = train['winPlacePerc']
    x = train.drop(['winPlacePerc', "Id"], axis=1)