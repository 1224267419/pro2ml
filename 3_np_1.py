import numpy as np

# a = []
# for i in range(10000000):
#     a.append(random.random())
# # 通过time方法，查看当前行的代弱运行一次所花费的时间
# current_time = time.time()
# sum1 = sum(a)
# print(time.time() - current_time)
# b = np.array(a)  # 转换为np数组
# current_time = time.time()
# sum2 = np.sum(b)
# print(time.time() - current_time)  # 用numpy速度快接近4倍


aa = np.array([[80, 89, 86, 67, 79],
               [78, 97, 89, 67, 81],
               [90, 94, 78, 67, 74],
               [91, 91, 90, 67, 69],
               [76, 87, 75, 67, 86],
               [70, 79, 84, 67, 84],
               [94, 92, 93, 67, 64],
               [86, 85, 83, 67, 80]])
print(aa.ndim)  # 维度
print(aa.shape)  # 形状
#
# arr = np.ones([4, 8, 3])  # 创建对应形状的数组
# arr1 = np.zeros_like(arr)  # 生成和arr同形状的全0数组
# arr2 = np.arange(10, 50, 3)  # 等差数列(起点,终点,间隔)
# arr3 = np.logspace(-10, 5, 3)  # 生成等比数列(实际生成数值为10^n),(起点,终点,元素数量)
# print(arr)
# print(arr1)
# print(arr2)
# print(arr3)
#
# arr4 = np.random.normal(1.75, 1, 1000000)  # 正态分布,(均值=1.75,标准差=1,num=1000000)
# plt.figure()
# plt.hist(arr4, 1000)  # 绘制直方图,bins：指定直方图条形的个数(数据分成多少组
# plt.show()
#
# arr5 = np.random.rand(3, 2, 3)  # 生成0,1均匀分布的数值,形状如参数所示
# print(arr5.mean())
#
# arr6=np.random.uniform(-1,1,100000000)#上下区间的均匀分布
# print(arr6.mean())
# plt.hist(arr6, 10000)  # 绘制直方图,bins：指定直方图条形的个数(数据分成多少组
# plt.show()

arr5 = np.random.rand(3, 2)
print(arr5.shape)
arr5.resize([6, 1])  # 改变原数组形状
print("resize后的形状" + str(arr5.shape))

print("reshape的形状" + str(arr5.reshape([1, 6])))  # 不改变原数组形状
print("reshape后的形状" + str(arr5.shape))

print(arr5.T.shape)  # 转置(不改变源数组
print(arr5.shape)
arr5.astype(np.int32)  # 修改数据类型
print(arr5.tostring())  # to系列可以帮助数据转换
a = np.array([[1, 6, 3, 4], [2, 3, 4, 5]])
print(np.unique(a))  # 去除重复元素,并含顺序排列
# print(a)  #不改变原数组a
