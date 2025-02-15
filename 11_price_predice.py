import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV  # 线性回归
from sklearn.linear_model import SGDRegressor  # 梯度下降
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 标准化
# 1.获取数据
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 数据准备
X = data
y = target


def model1(X, y):
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化数据
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.fit_transform(X_test)
    # 实例化线性回归模型
    model = LinearRegression()  # 默认使用正规方程而非梯度下降法

    # 模型训练
    model.fit(X_train, y_train)
    model=joblib.load("data.pkl")#模型读取
    # 模型预测
    y_pred = model.predict(X_test)

    # 计算均方误差
    mse = mean_squared_error(y_pred, y_test)
    print("LR均方误差：", mse)
    # print("LR模型系数是", model.coef_)
    # 结果可视化
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('actual_price')
    plt.ylabel('predict_price')
    plt.title('liner')
    plt.show()
    joblib.dump(model,"./data.pkl")#模型保存


def model2(X, y):
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化数据
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.fit_transform(X_test)
    # 实例化线性回归模型
    model = SGDRegressor(max_iter=1000, learning_rate="constant",
                         eta0=0.001)  # 梯度下降法的线性回归,有许多参数可以调节,包括学习率,是否自动降低学习率等

    # 模型训练
    model.fit(X_train, y_train)

    # 模型预测
    y_pred = model.predict(X_test)
    # 计算均方误差
    mse = mean_squared_error(y_pred, y_test)
    print("梯度下降线性回归模型均方误差：", mse)
    # print("梯度下降线性回归模型系数是", model.coef_)

    # 结果可视化
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('actual_price')
    plt.ylabel('predict_price')
    plt.title('liner')
    plt.show()


def model3(X, y):
    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化数据
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.fit_transform(X_test)
    # 实例化线性回归模型
    model = Ridge(alpha=1.0)  # 岭回归
    model = RidgeCV(alphas=(0.001, 0.01, 0.1, 10, 100))  # 网格搜索岭回归

    # 模型训练
    model.fit(X_train, y_train)

    # 模型预测
    y_pred = model.predict(X_test)
    # 计算均方误差
    mse = mean_squared_error(y_pred, y_test)
    print("岭回归模型均方误差：", mse)
    # print("岭回归模型系数是", model.coef_)

    # 结果可视化
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('actual_price')
    plt.ylabel('predict_price')
    plt.title('liner')
    plt.show()


if __name__ == '__main__':  # 对比三个模型的均方差
    model1(X, y)
    # model2(X, y)
    # model3(X, y)
