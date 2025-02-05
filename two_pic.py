import matplotlib.pyplot as plt
import numpy as np
import random

# 创建一些测试数据
x = range(60)
y_shanghai = [random.uniform(5, 8) for i in x]  # 在5-8随机生成小数
y_beijing = [random.uniform(0,3) for i in x]  # 在0-3随机生成小数

# 一张图片放两个坐标系
fig, ax = plt.subplots(nrows=1,ncols=2)

ax[0].plot(x,y_beijing)#传出的画布用数组即可调用,和之前的方法无异
ax[1].plot(x,y_shanghai)
plt.show()