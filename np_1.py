import random
import time
import numpy as np
a=[]
for i in range(10000000):
    a.append(random.random())
#通过time方法，查看当前行的代弱运行一次所花费的时间
current_time = time.time()
sum1=sum(a)
print(time.time()-current_time)
b=np.array(a)#转换为np数组

current_time = time.time()
sum2=np.sum(b)
print(time.time()-current_time)#用numpy速度快接近4倍
aa=np.array([[80,89,86,67,79],
    [78,97,89,67,81],
    [90,94,78,67,74],
    [91,91,90,67,69],
    [76,87,75,67,86],
    [70,79,84,67,84],
    [94,92,93,67,64],
    [86,85,83,67,80]])
print(aa.ndim)#维度
print(aa.shape)#形状