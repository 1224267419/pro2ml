import pandas as pd
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # 交叉验证网格搜索
from sklearn.preprocessing import StandardScaler  # 标准化

# 1.获取数据
data = pd.read_csv("./train.csv")
# print("\n",data.head)  # 查看数据机构和整体
# print("\n",data.describe())  # 数据属性(做了一些简单运算,包括最大最小平均值,中位数和四分数
# print("\n",data.shape)

# 2.数据处理
# 2.1.缩小数据范围(这里为了简化数据)
data = data.query("x>2.0 & x<2.5 & y>2.0 & y<2.5")
print(data.shape)

time = pd.to_datetime(data["time"], unit='s')
print(time)  # dtype: datetime64[ns]
# 2.2 把时间戳转换为常用的日期格式,以秒为单位,转换后的时间为列表
time = pd.DatetimeIndex(time)  # 生成时间戳索引DatetimeIndex类型而不是之前的普通pd.series
# print(time)
# print(time.hour)
# print(time.day)
data["day"] = time.day
data["hour"] = time.hour
data["weekday"] = time.weekday
print(data.head())
# 2.3 去掉签到较少的地方
place_count = data.groupby("place_id").count()
# 通过分组聚合, 记录签到地点出现次数,每一列都被修改为计数,因此任选一列即可
# place_count["row_id"]>3 即出现次数大于3的行为true,小于为false
place_count = place_count[place_count["row_id"] > 3]  # 仅保留出现次数＞3的项
print(place_count.head())
data = data[data['place_id'].isin(place_count.index)]  # 仅保留place_count.index索引中存在的项
# print(data.shape)  # 减少了10000项

x = data[["x", "y", "accuracy", "hour", "day", "weekday"]]  # 特征值
y = data["place_id"]  # 目标值

# 2.4 分割数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)  # 划分数据集
transfer = StandardScaler()
# 标准化数据
x_train = transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

# 5.学习
estimator = KNeighborsClassifier()  # 实例化
temp=[i for i in range(10)]
param_dict = {"n_neighbors": temp[1::2]}  # 字典传入要验证的参数
# print(temp[1::2])
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=10,n_jobs=-1)#交叉验证
#n_jobs用于调用多cpu,-1即调用所有cpu
# param_grid为待被筛选的超参数字典； cv为交叉验证的折数,estimator为实例化后的算法模型对象
estimator.fit(x_train, y_train)  # 训练模型

print('准确率为', estimator.score(x_test, y_test))#模型评估
print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
print("最好的参数模型：\n", estimator.best_estimator_)
print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)  # 所有结果