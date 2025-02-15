from sklearn.linear_model import  LinearRegression #线性回归
from sklearn.linear_model import  SGDRegressor #梯度下降


x=[[80,86],
[82,80],
[85,78],
[90,90],
[86,82],
[82,90],
[78,80],
[92,94]]
y=[84.2,80.6,80.1,90,83.2,87.6,79.4,93.4]
estimator=LinearRegression()
estimator.fit(x,y)#训练
print(estimator.coef_)#输出训练的回归系数
print(estimator.predict([[100,80]]))#predict允许同时预测多组数据,因此需要加两层[]