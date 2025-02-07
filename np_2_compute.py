import numpy as np
#
# score = np.random.randint(40, 100, (10, 5))  # 40-100分均匀分布,10个人每人5科成绩
# print(score)
# print(score > 60)  # 逻辑判断返回的数值为bool类型
# print(np.all(score > 60))  # and,这个函数是用于判断是否所有元素为true
# print(np.any(score > 60))  # or,这个函数是用于判断是否存在元素为true
# print(np.where(score > 60, 1, 0))  # 类比三目运算符,符合的写1,不符合写0
# print(np.where(np.logical_and(score > 60, score < 90), 1, 0))  # np.logical_提供常见的与或非方法,从而实现逻辑组合
#
# print(score)#矩阵
# print(np.max(score))#最大值
# print(np.argmax(score))#输出最大值的坐标,下同理
# print(np.max(score,axis=0))#0维(列)最大值
# print(np.max(score,axis=1))#1维(行)最大值
#
# print(score + 5)  # 允许整个矩阵进行运算,结果为每个元素单独运算后的结果(广播机制
# arr1=np.array([[0],[1],[2],[3]])
# arr2=np.array([1,2,3])
# # print(arr1.shape)
# print(arr1+arr2)

#矩阵运算
a= np.array(
[[80,86],
[82,80],
[85,78],
[90,90],
[86,82],
[82,90],
[78,80],
[92,94]])
b=np.array([[0.7],[0.3]])
print(np.dot(a,b))
print(np.matmul(a,b))