from sklearn.neighbors import KNeighborsClassifier

x=[[0],[1],[2],[3]] #特征
y=[0,0,1,1] #类别
estimator=KNeighborsClassifier(n_neighbors=1)#实例化类,使用最近的1个特征进行拟合
estimator.fit(x,y)#特征和分类拟合
print(estimator.predict([[5]])) #根据特征预测结果