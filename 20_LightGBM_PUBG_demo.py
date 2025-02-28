

import numpy as np
import lightgbm as lgb
from gmpy2 import random_state
from lightgbm import early_stopping
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
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
from tensorflow.python.data.experimental.ops.optimization import model
import seaborn as sns


def data_preprocessing1():#数据处理1


    df = pd.read_csv("pubg_data/train_V2.csv")

    print(df.head())
    print(df.tail())
    print(df.describe())#常用信息
    print(df.info())#各列信息
    print(np.unique(df['matchId']).shape)#比赛数(排除了重复项
    print(np.unique(df['groupId']).shape)#队伍数(排除了重复项

    print(df.isnull())
    print(np.any(df.isnull()))
    #排除空白项
    print(df[df['winPlacePerc'].isnull()])
    train=df

    train.drop(2744604, inplace=True)
    train.shape

    # 显示每场比赛参加人数
    # transform的作用类似实现了一个一对多的映射功能，把统计数量映射到对应的每个样本上
    count = train.groupby('matchId')['matchId'].transform('count')

    train['playersJoined'] = count
    #从而得出每场比赛有多少人参与
    count.count()

    train.head()

    np.any(train.isnull())

    train['playersJoined'].sort_values().head() #按升序排序,发现有的比赛只有个位数的人参加

    plt.figure(figsize=(20,10))
    sns.countplot(train['playersJoined'])
    plt.title('playersJoined')
    plt.grid()
    plt.show()

    # 限制每局开始人数大于等于75，再进行绘制。
    # 再次绘制每局参加人数的直方图
    plt.figure(figsize=(20,10))
    sns.countplot(train[train['playersJoined']>=75]['playersJoined'])
    plt.title('playersJoined')
    plt.grid()
    plt.show()
    #75人以上参加比赛的数量比较可观
    return train[train['playersJoined']>=75]

def data_preprocessing2(train): #删除不合理数据

    # 变量合成创建新变量“healsandboosts”
    train['healsandboosts'] = train['heals'] + train['boosts']
    train[["heals", "boosts", "healsandboosts"]].tail()
    # 此处我们把特征：heals(使用治疗药品数量)和boosts(能量、道具使用数量)合并成一个新的变量，命名：”healsandboosts“

    '''
    4.2.4.1 异常值处理：删除有击杀，但是完全没有移动的玩家
    异常数据处理：

    一些行中的数据统计出来的结果非常反常规，那么这些玩家肯定有问题，为了训练模型的准确性，我们会把这些异常数据剔除

    通过以下操作，识别出玩家在游戏中有击杀数，但是全局没有移动；

    这类型玩家肯定是存在异常情况（挂**），我们把这些玩家删除。'''
    # 创建新变量，统计玩家移动距离
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
    # 创建新变量，统计玩家是否在游戏中，有击杀，但是没有移动，如果是返回True, 否则返回false
    train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

    np.any(train['killsWithoutMoving'].isnull())

    train[train['killsWithoutMoving'] == True].head()
    """
    找出train['killsWithoutMoving']为True的块,异常数据自然要删除
    """
    train[train['killsWithoutMoving'] == True].index

    train.shape
    train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
    train.drop(train[train['roadKills'] == True].index, inplace=True)

    train
    """
    行数减少,上述的异常数据(驾车杀敌过多和无移动杀敌)的行已被删除
    """
    train[train['kills'] > 30].shape  # 杀敌数过多

    train.drop(train[train['kills'] > 30].index, inplace=True)

    train.shape

    # 如果一个玩家的击杀爆头率过高，也说明其有问题
    # 创建变量爆头率
    train['headshot_rate'] = train['headshotKills'] / train['kills']
    train['headshot_rate'] = train['headshot_rate'].fillna(0)  # 可能有除以0的情况出现,这些值均为NaN

    # 绘制爆头率图像
    plt.figure(figsize=(12, 4))
    sns.distplot(train['headshot_rate'], bins=10, kde=False)
    plt.show()
    """
    显然,右侧存在一些异常的爆头率数据,将其剔除
    """
    train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape

    train.drop(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].index, inplace=True)

    train.shape

    # 绘制图像
    plt.figure(figsize=(12, 4))
    sns.distplot(train['longestKill'], bins=10, kde=False)
    plt.show()
    # 找出最远杀敌距离大于等于1km的玩家
    train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
    train.shape

    # 异常值处理：删除关于运动距离的异常值
    train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
    train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
    train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)

    # 异常值处理：删除使用治疗药品数量异常值
    train.drop(train[train['heals'] >= 40].index, inplace=True)

    # 关于比赛类型，共有16种方式
    train['matchType'].unique()
    '''
    array(['squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo',
           'normal-squad-fpp', 'crashfpp', 'flaretpp', 'normal-solo-fpp',
           'flarefpp', 'normal-duo-fpp', 'normal-duo', 'normal-squad',
           'crashtpp', 'normal-solo'], dtype=object)
           '''

    # 对matchType进行one_hot编码
    # 通过在后面添加的方式,实现,赋值并不是替换
    train = pd.get_dummies(train, columns=['matchType'])

    # 关于groupId,matchId这类型数据，也是类别型数据。但是它们的数据量特别多，如果你使用one-hot编码，无异于自杀。
    # 在这儿我们把它们变成用数字统计的类别型数据依旧不影响我们正常使用。
    # 将字符串型的数据转换为分类数据,下述两行将数据变成从0开始的数字
    train['groupId'] = train['groupId'].astype('category')
    train["groupId_cat"] = train["groupId"].cat.codes
    train["groupId_cat"].head()

    # 转换match_id(同上
    train['matchId'] = train['matchId'].astype('category')

    train['matchId_cat'] = train['matchId'].cat.codes

    # 删除之前列
    train.drop(['groupId', 'matchId'], axis=1, inplace=True)

    # 查看新产生列
    train[['groupId_cat', 'matchId_cat']].head()

    # 确定特征值和目标值
    x = train.drop(["winPlacePerc", "Id"], axis=1)  # 特征值

    y = train['winPlacePerc']  # target
    y.head()

    x.shape
    """
    模型训练
      1.RF模型
    """
    # 数据分割
    return train_test_split(train, y, test_size=0.2)

def random_forest_train1(df,X_train, X_valid, y_train, y_valid):#随机森林
    # 模型训练
    from sklearn.ensemble import RandomForestRegressor
    m1 = RandomForestRegressor(n_estimators=40,
                               min_samples_leaf=3,
                               max_features='sqrt',
                               n_jobs=-1)

    m1.fit(X_train, y_train)

    y_pre = m1.predict(X_valid)
    print("默认随机森林的准确率为",m1.score(X_valid, y_valid))  # 模型准确率约为0.923
    print("默认随机森林的mae为",mean_absolute_error(y_true=y_valid, y_pred=y_pre))  # mae=0.060


    # 查看特征值在当前模型中的重要程度
    print(m1.feature_importances_)
    imp_df = pd.DataFrame({"cols": df.columns, "imp": m1.feature_importances_})
    imp_df = imp_df.sort_values("imp", ascending=False)  # 排序
    # 绘制特征重要性程度图，仅展示排名前二十的特征
    imp_df[:20].plot('cols', 'imp', figsize=(20, 8), kind='barh')
    plt.show()
    return imp_df
    print()


def random_forest_train2(X_train, X_valid, y_train, y_valid,imp_df):
    #通过之前预训练的随机森林,找出比较重要的特征重要程度imp_df


    # 保留比较重要的特征
    to_keep = imp_df[imp_df.imp > 0.005].cols
    print('Significant features: ', len(to_keep))
    print(to_keep)

    # 由这些比较重要的特征值，生成新的df
    print(train[to_keep].shape)  # 保留了16个特征

    # 新模型
    m2 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',
                               n_jobs=-1)
    # 模型训练
    m2.fit(X_train, y_train)

    # 模型评分
    y_pre = m2.predict(X_valid)

    print("排除部分项后随机森林的准确率为",m2.score(X_valid, y_valid))
    print("排除部分项后随机森林的mae为",mean_absolute_error(y_true=y_valid, y_pred=y_pre))
    '''
    原49个特征的准确率=0.9230867436715275
    原49个特征的mae=0.06049533416907304
    新结果比原结果更好,显然防止了过拟合
    '''


if __name__ == '__main__':

    train=data_preprocessing1() # 数据预处理1
    X_train, X_valid, y_train, y_valid=data_preprocessing2(train)# 数据预处理2
    imp_df=random_forest_train1(train,X_train, X_valid, y_train, y_valid) # 随机森林训练和修正
    random_forest_train2(X_train, X_valid, y_train, y_valid,imp_df) # 根据训练结果修正数据后的随机森林训练和修正



