# pro2ml 

###### 用matplot画图

[huatu.py](huatu.py) 简单折线图

![image-20250201234708725](README.assets/image-20250201234708725.png)



绘制基本图像 [basic_pic.py](basic_pic.py) 

一张图片同时绘制两个坐标图 [two_pic.py](two_pic.py) 

剩余部分见[教程 ](https://www.runoob.com/matplotlib/matplotlib-line.html)



numpy 

[np_1.py](np_1.py) 常用numpy数组的构建

[np_2_compute.py](np_2_compute.py)numpy数组的运算和统计方法



下图为广播机制,满足下列其中一个条件即可触发广播,扩充短的维度为长的

1数组的某一维度等长
2其中一个数组的某一维度为1。



![image-20250208005014542](README.assets/image-20250208005014542.png)

对于np.dot和np.matmul,都是矩阵乘法区别在于

- *np.dot*可以处理一维数组和标量的乘法，而*np.matmul*不能。
- 在处理二维数组时，*np.dot*和*np.matmul*的结果相同，都是矩阵乘积。
- *np.matmul*在处理高维数组时更加灵活，它可以处理多个数组的矩阵乘积，并且可以自动广播数组以匹配维度。

如果你需要计算标量或一维数组的点积，或者是两个二维数组的矩阵乘积，*np.dot*是合适的选择。如果你需要处理高维数组的矩阵乘积，并且希望利用自动广播的特性，那么*np.matmul*将是更好的选择

# [TODO]()
