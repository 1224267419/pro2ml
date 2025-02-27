import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
# 用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

# ⽣成数据
x = np.array(list(range(1, 11))).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05])
 # 训练模型
model1 = DecisionTreeRegressor(max_depth=1) # 二元回归
model2 = DecisionTreeRegressor(max_depth=3) # 决策树模型
#决策树深度越大,模型越复杂,降低决策树深度可以防止过拟合
model3 = linear_model.LinearRegression()  # 线性回归模型
model1.fit(x, y)
model2.fit(x, y)
model3.fit(x, y)

# 模型预测
X_test = np.arange(0.0, 10.0, 0.01).reshape(-1, 1) # ⽣成1000个数,⽤于预测模型
# print(X_test.shape)#必须要变成(1000, 1)才能画图
y_1 = model1.predict(X_test)
y_2 = model2.predict(X_test)
y_3 = model3.predict(X_test)

# 结果可视化
plt.figure(figsize=(10, 6), dpi=100)

#点图(画出训练集的数据点

plt.scatter(x, y, label="data")

#折线图
plt.plot(X_test, y_1,label="max_depth=1")
plt.plot(X_test, y_2, label="max_depth=3")
plt.plot(X_test, y_3, label='liner regression')

plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")

plt.legend()
plt.show()
