import random

import matplotlib.pyplot as plt

x = range(60)
# print(x)
y_shanghai = [random.uniform(15, 18) for i in x]  # 在15-18随机生成小数
# print(y_shanghai)
plt.figure(figsize=(20, 8), dpi=80)  # 绘制画布
plt.plot(x, y_shanghai)  # 绘制x,y轴的点
# x and y must have same first dimension, but have shapes (1,) and (60,),画图的x和y要有相同的第一维
plt.show()  # 展示画布
