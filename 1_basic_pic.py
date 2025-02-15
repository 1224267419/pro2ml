import random

import matplotlib.pyplot as plt
from pylab import mpl  # 字体修改

# 使用黑体字显示,从而显示中文
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 生成虚拟数据
x = range(60)
# print(x)
y_shanghai = [random.uniform(5, 8) for i in x]  # 在5-8随机生成小数
y_beijing = [random.uniform(0,3) for i in x]  # 在0-3随机生成小数

# print(y_shanghai)

# 坐标轴文字
x_lebal_ticks = ["1点{}分".format(i) for i in x]  # 添加x轴y轴刻度
y_ticks = [i for i in range(40)]
print(x_lebal_ticks)
print(y_ticks)

# # 坐标轴
# plt.xticks(x[::5], x_lebal_ticks[::5])
# # 坐标轴从0-60以5为间隔,第一个参数是坐标轴对应数值,下一个参数是对应位置的实际文字
# plt.yticks(y_ticks[::5])

plt.figure()  # 绘制画布
x2 = [i for i in range(60)]
# print(x2)

plt.plot(x, y_shanghai,color="b")  # 绘制绘制x,y轴的折线图
plt.plot(x,y_beijing,color="g")  # 在一幅图同时绘制两个x,y轴的折线图

# plt.plot(x2, y_shanghai)  # 绘制和上行等价
# x and y must have same first dimension, but have shapes (1,) and (60,),画图的x和y要有相同的第一维

# 显示网格线,ls为网格线类型
plt.grid(True, ls=':')

font_dict = dict(fontsize=16,  # 用一个字典调整字体的各个样式,要用的时候搜索一下
                 weight='light',
                 style='italic',
                 )

# 为坐标轴提供标题
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("总标题", fontdict=font_dict)

# plt.savefig("way.png")#保存图片
# show会释放资源,save放在show后会保存空图片
plt.show()  # 展示画布
