# pro2ml 

###### 用matplot画图

 [1_basic_pic.py](1_basic_pic.py) 简单折线图

![image-20250201234708725](README.assets/image-20250201234708725.png)



一张图片同时绘制两个坐标图 [two_pic.py](2_two_pic.py) 

剩余部分见[教程 ](https://www.runoob.com/matplotlib/matplotlib-line.html)



###### numpy 

[np_1.py](3_np_1.py) 常用numpy数组的构建

[np_2_compute.py](4_np_2_compute.py)numpy数组的运算和统计方法



下图为广播机制,满足下列其中一个条件即可触发广播,扩充短的维度为长的

1数组的某一维度等长
2其中一个数组的某一维度为1。



![image-20250208005014542](README.assets/image-20250208005014542.png)

对于np.dot和np.matmul,都是矩阵乘法区别在于

- *np.dot*可以处理一维数组和标量的乘法，而*np.matmul*不能。
- 在处理二维数组时，*np.dot*和*np.matmul*的结果相同，都是矩阵乘积。
- *np.matmul*在处理高维数组时更加灵活，它可以处理多个数组的矩阵乘积，并且可以自动广播数组以匹配维度。

如果你需要计算标量或一维数组的点积，或者是两个二维数组的矩阵乘积，*np.dot*是合适的选择。如果你需要处理高维数组的矩阵乘积，并且希望利用自动广播的特性，那么*np.matmul*将是更好的选择



### ML概念

通过数据+算法实现功能,而不是仅靠基于规则的学习(if..else),让程序自己从数据中提取特征

##### 监督学习

给出特征值(feature)和目标值(labal),训练能完成分类或回归的模型

###### 分类:结果是已知的离散值

###### 回归:结果为连续值

##### 无监督学习

所给数据没有目标值,只有特征值

###### 聚类:为已有数据进行分类

###### 数据降维:清除噪声,压缩数据

##### 半监督学习

利用**少量**已有**标签**的数据训练模型
再利用该模型套用未标记数据
最后进行人工优化

###### 纯半监督学习:未标记样本不是待测数据

###### 直推学习(transductive learning):未标记样本就是待测数据

##### 强化学习RL

强化学习不像无监督学习那样完全没有学习目标，又不像监督学习那样有非常明确的目标（即
lbel),强化学习的目标一般是变化的、不明确的，甚至可能不存在绝对正确的标签
一般强化学习通过设置一个Agent,通过与环境的交互来获得奖励或惩罚,目标是长期利益最大化

##### 欠拟合

训练集上表现差:一般是模型过于简单
下述方法可以解决或者缓解欠拟合
 ①添加新特征,特征添加二次或三次项
 ②复杂化模型
 ③减少正则化
![image-20250211171633258](README.assets/image-20250211171633258.png)

红字的欠拟合和过拟合是指上面的v字线

##### 过拟合

 ①清洗数据
 ②增大训练数据量

######  ③正则化(减少高次项)

###### ·L2正则化

。作用：可以使得其中一些W的都很小，都接近于0，削弱某个特征的影响
。优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象

###### 。Ridge岭回归(常用<a name="岭回归"></a>

$$
J(\theta) = \text{RSS} + \alpha \sum_{i=1}^{n} \theta_i^2
$$

RSS为残差,从而减少回归参数

·L1正则化
。作用：可以使得其中一些W的值直接为0，删除这个特征的影响
。LASSO回归(特征较少时使用
$$
J(\theta) = \text{RSS} + \alpha \sum_{i=1}^{n} |\theta_i|
$$

###### 弹性网络Elastic Net(常用

$$
J(\theta) = \text{MSE}(\theta) + \alpha \sum_{i=1}^{n} |\theta_i| + \frac{1-\rho}{2} \alpha \sum_{i=1}^{n} \theta_i^2
$$

通过参数r控制L1和L2的比例

 ④dropout
 ⑤早停




##### 标准化

$$
X_i=\frac{x_i - mean}{\sigma_i}
$$

标准化后数据均值为0,标准差为1,

```python
from sklearn.preprocessing import StandardScaler
transformer =StandardScaler()
data=transformer.fit_transform(data)
```

可以通过上述代码实现特征归一化

##### 特征提取<a id="特征提取"></a>

任意数据（如文本或图像）转换为可用于机器学习的数宇特征(如图片变为rgb或yuv矩阵)





#### sklearn

一个机器学习包,提供 **分类,回归,聚类,降维(特征工程),模型选择,调优** 六大模块

#### 1.1 K-近邻算法(KNN)概念 [k_neighbor.py](5_k_neighbor.py) 

K Nearest Neighbor算法又叫KNN算法，这个算法是机器学习里面一个比较经典的算法，总体来说KNN算法
是相对比较容易理解的算法

##### ·定义

样本类别定义为k个距离最近的样本中的众数的类别
KNN使用欧氏距离



##### 距离度量

###### 欧氏距离：

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
$$

###### 曼哈顿距离

![image-20250208104252227](README.assets/image-20250208104252227.png) 

黄色,蓝色,红色为曼哈顿距离,公式如下
$$
d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{n} \left| x_i - y_i \right|
$$

###### 切比雪夫距离



$$
d(\mathbf{x}, \mathbf{y}) = \max_{1 \leq i \leq n} \left| x_i - y_i \right|
$$

###### 闵可夫斯基距离

$$
d(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} \left| x_i - y_i \right|^p \right)^{\frac{1}{p}}
$$

p=1时为曼哈顿距离

p=2时为欧氏距离

P→∞时为切比雪夫距离

上述距离存在一些问题,由于数据量纲不同,直接套用距离显然不太合适

###### 标准化欧氏距离公式

标准化欧氏距离在计算欧氏距离时考虑了各个特征的尺度差异，其公式为：

$$
d(\mathbf{x}, \mathbf{y}) = \sqrt{ \sum_{i=1}^{n} \left(\frac{x_i - y_i}{\sigma_i}\right)^2 }
$$

$$
\sigma_i 为第 i 个特征的标准差。
$$

这种距离度量在特征具有不同单位或量纲时非常有用，通过先对各个特征进行标准化，再计算欧氏距离，从而消除了不同尺度带来的影响。



###### 余弦距离

$$
\text{cosine similarity} = \frac{\sum_{i=1}^{n} X_i Y_i}{\sqrt{\sum_{i=1}^{n} X_i^2}\sqrt{\sum_{i=1}^{n} Y_i^2}}
$$



还有汉明距离,杰卡德距离,马氏距离...



##### 连续属性与离散属性

若属性值之间存在序关系，则可以将其转化为连续值，例如：身高属性“高”“中等“矮”，可转化为
{1,0.5,0}
			闵可夫斯基距离可以用于有序属性，
若属性值之间不存在序关系，则通常将其转化为向量的形式，例如：性别属性"男"“女”，可转化为
{0,1}

##### k值的选择

**近似误差**:训练集误差,太小可能过拟合
**估计误差** :测试集误差,越低越好

·K值过小：
。容易受到异常点的影响(过拟合
k值过大：
。受到样本均衡的问题,样本分布构建的模型有问题(模型太简单
取极端例子,k=n时任何预测均为样本的众数,失去了判断能力

一般而言,k值的选择通过交叉验证法找出最优k值

##### kd树

kd树在维度<20时效率较高,更高维度的数据可以使用ball tree

目的:优化k临近算法,提高knn的搜索效率,减少距离计算次数
原理:AB远,BC近,则AC远,从而降低运算复杂度

将时间复杂度从O(D*N^2)降低至O(DNlogN)

构造:通过对k维空间进行切分(超平面),用**中位数**递归生成子节点,产生一颗**平衡二叉树**
一般而言,选择分割的维度时会选择数据较为分散的维度(用**方差**来判断)
构建时**相邻两层的划分维度必定不相同**(正交划分)

搜索:从根节点向下遍历,找出最终归属的区域,与对应的节点求出距离即可(logN)

下面左图为辅助理解超平面,右图为树结构,⭐为要查找的点

**不跨域搜索:**

![](README.assets/image-20250208163847312.png)

不跨域是因为(5,4)到(4,7)的距离＞最小距离,因此无需查找另外一侧

**跨域搜索:**

![image-20250208163935916](README.assets/image-20250208163935916.png)

跨域搜索(2,3)是因为当前最小距离>(5,4)到(2,3)的距离

上图所说的画圆只是辅助理解,实际上还是求两点间距离

搜索过程:
①从根节点开始,比较**待查询节点和分裂节点的分裂维的值**(小于进左子树,大于进右子树
②**沿着搜索路径找到最近邻近似点**
③**回溯搜索路径**,如果存在可能比当前距离更近的点,则跳转到其他子节点空间
④重复上述过程直至**搜索路径**为空



##### 损失函数

用于衡量数据拟合程度
**线性拟合**的损失函数一般用最小二乘法

损失函数公式：(欧式距离的平方)
$$
J(w) = \sum_{i=1}^{m} (h(x_i) - y_i)^2
$$
正则化,标准化数据  [Standard.py](6_Standard.py) 



#####  [knn_2.py](7_knn_2.py) 使用knn对提供的数据集进行拟合和划分

在程序中存在两个函数fit_transform()和transform()

**`fit_transform()` 方法：**

- **功能**：首先对数据进行拟合（`fit`），计算所需的统计量（如均值、标准差等），然后立即对数据进行转换（`transform`）。

  在上述代码中，`fit_transform()` 首先计算训练数据 `X_train` 的均值和标准差，然后使用这些统计量对 `X_train` 进行标准化处理。

**`transform()` 方法：**

- **功能**：使用在 `fit` 阶段计算得到的统计量，对新的数据进行转换。
- **使用场景**：用于测试数据集或任何新的数据集，以**确保训练集和测试集使用相同的转换标准**。

因此程序中

```python
x_train = transfer.fit_transform(x_train)
# x_test = transfer.fit_transform(x_test)
x_test = transfer.transform(x_test)#因为都使用标准化方法,因此上下二式子等价(相同fit)
```

最后两行等价

##### 总结

###### 优点：

。简单有效
。重新**训练的代价低**
。**适合类域交叉样本**(不同类别的样本存在交叉或重叠的情况)
	·KNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，KNN方法较其他方法更为适合。
。**适合大样本自动分类**
	·该算法比较适用于样本容量比较大的类域的自动分类，而那些样本容量较小的类域采用这种算
法比较容易产生误分。

###### ·缺点：

。惰性学习
·KNN算法是懒散学习方法(lazy learning,**基本上不学习**)，一些积极学习的算法要快很多
。类别**评分不是规格化**,没有分类概率
。输出**可解释性不强**
。对**不擅长不均衡的样本**

##### 交叉验证

数据分成k组,每次取1组作为验证集,其他作为测试集,求准确率;不同测试集验证k次后结果求平均即为交叉验证的准确率

##### 网格搜索

手动设置超参数过于复杂,因此通过对模型预设的超参数组合进行交叉验证,选出最优参数组合较为合理,下面是网格搜索代码

 [Grid_Search.py](8_Grid_Search.py)

```python
estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5)
#param_grid为待被筛选的超参数字典； cv为交叉验证的折数,estimator为实例化后的算法模型对象
```

准确率和最佳结果不一样,是因为交叉验证使用了测试集作为训练集的一部分,因此准确率较高
(代码26,27行

##### 用knn算法参加比赛

 [比赛链接](https://www.kaggle.com/competitions/facebook-v-predicting-check-ins/data),数据已在本文件夹中 
[Fcaebook_example.py](9_Fcaebook_example.py) 实际处理,实际还是使用之前的函数



### 线性回归linear regression



###### 线性关系

公式如下,其中w,x均为列向量
$$
h(x)==w_1 x_1 +w_2 {x_2}+w_3 {x_3}+ b=w^T x + b
$$

###### 非线性关系

存在高次项的是非线性关系,例子公式如下
$$
h(x)=w_1 x_1 +w_2 {x_2}^2+w_3 {x_3}^3+ b \\
其中
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
\mathbf
\qquad{w} =
\begin{bmatrix}
w_1 \\
w_2 \\
w_3
\end{bmatrix}
$$

对于线性回归,我们常用的**损失函数**如下:
$$
J(w) = (h(x_1) - y_1)^2 + (h(x_2) - y_2)^2 + \cdots + (h(x_m) - y_m)^2 \\
= \sum_{i=1}^{m} (h(x_i) - y_i)^2
$$

##### 损失函数

通过计算一个数值，表示模型预测的准确性或误差大小。在训练过程中，模型的目标是通过调整其参数来最小化损失函数的值，从而提高预测的准确性

###### 损失函数决定了算法的思想

##### 损失函数优化

###### 1.正规方程(t通过矩阵运算得到)

$$
w = (X^T X)^{-1} X^T y
$$

假设最优解存在,那上述公式必然能**一步求得最优结果**(直接求解w)

缺点:特征过于复杂时计算量大(O(n^3))且得不到结果

##### 2.梯度下降(SGD)

梯度:函数上升速率最快的方向

公式如下:
$$
\theta_{i+1} = \theta_{i}-\alpha \cdot J'(\theta_i)\\
\alpha为学习率
$$
学习率太大:错过最低点
学习率太小:走不到最低点(卡在极小值)

对于多个参数,线性方程的损失函数如下所示
$$
 J(\theta_0, \theta_1, \ldots, \theta_n) = \frac{1}{2m} \sum_{j=0}^m \left( h_\theta \left( x_0^{(j)}, x_1^{(j)}, \ldots, x_n^{(j)} \right) - y_j \right)^2 
$$

$$
更新后
\theta_{i+1} =\theta_{i} - \alpha*
\frac{\partial}{\partial \theta_{i}} J(\theta_{0}, \theta_{1}, \ldots, \theta_{n})

\\=\theta_{i} - \alpha \cdot \frac{1}{m} \sum_{j=0}^{m} \left( h_{\theta} \left( x_{0}^{(j)}, x_{1}^{(j)}, \ldots, x_{n}^{(j)} \right) - y_{j} \right) x_{i}^{(j)}
$$

不断迭代θ_i直至损失函数J<一个值

###### 全梯度下降FG

计算所有样本误差,再进行梯度下降,中途不能添加样本

###### 随机梯度下降SG

随机选择单个样本,可以有噪声影响

###### mini-batch

选择一定批量来做下降,上述二者结合体

###### 随机平均梯度下降算法SAG

使用前n-1个梯度和当前梯度的均值,性能高,收敛效果好

```python
from sklearn.linear_model import  LinearRegression #线性回归
from sklearn.linear_model import  SGDRegressor #梯度下降
```

常用函数

 [11_price_predice.py](11_price_predice.py) 训练波士顿房价的线性模型,数据获取部分从网上解决

```python
model = LinearRegression()  # 默认使用正规方程而非梯度下降法
model2 = SGDRegressor(max_iter=1000,learning rate="constant",eta0=0.001)  # 梯度下降法的线性回归,有许多参数可以调节,包括学习率,是否自动降低学习率等
```

两个模型的实例化,SGD有丰富的参数可以调节

##### [岭回归的定义](#岭回归)

为[11_price_predice.py](11_price_predice.py)列出了三个函数,对比了不同训练方法得出的均方差,

```python
from sklearn.linear_model import Ridge, RidgeCV  # 线性回归
```

引入岭回归和支持网格搜索的岭回归(自带最优参数





#### 模型的保存和加载

[11_price_predice.py](11_price_predice.py)文件中存在

```python
model=joblib.load("data.pkl")#模型读取
joblib.dump(model,"./data.pkl")#模型保存
```

### 逻辑回归(Logistic Regression)

分类模型,用于类别的判断

输入:线性回归的结果

##### 激活函数:

sigmoid函数
$$
\sigma(x) = \frac{1}{1 + e^{-h(w)}}= \frac{1}{1 + e^{-W^Tx}}
$$
通过这个激活函数返回一个(0,1)之间的值,可以用于二分类,下图为sigmoid的图形,常用于二分类
![image-20250212013857407](README.assets/image-20250212013857407.png)

**激活函数决定了神经网络的输出形式**，而**损失函数**需要与这种输出形式相匹配，以正确**衡量预测与真实值之间的差异**。

- **分类任务**：
  - **激活函数**：输出层通常使用 **Softmax**（多分类）或 **Sigmoid**（二分类），将输出映射为概率分布。
  - **损失函数**：对应使用**对数似然损失** ,也称**交叉熵损失（Cross-Entropy Loss）**，$\text{Cross-Entropy Loss} = - \sum_{i=1}^{K} y_i \log(p_i)$直接衡量概率分布的差异。
  - **数学优势**：Softmax + 交叉熵的组合在反向传播时梯度计算更高效（梯度简化），避免了数值不稳定性。 
- **回归任务**：
  - **激活函数**：输出层通常使用 **线性激活函数**（无激活），直接输出连续值。
  - **损失函数**：常用 **均方误差（MSE）** 或 **平均绝对误差（MAE）**，衡量预测值与真实值的距离。

````
sklearn.linear_model.LogisticRegression(solver='liblinear',penalty='12',C= 1.0)
````

上述代码为逻辑回归的API,相当于`SGDClassifier(loss="log",penalty="")`,即用交叉熵作为损失函数的随机梯度下降(SAG)

`solver='liblinear'`仅用于one-versus-rest问题(多分类转化为多个二分类),其他参数见代码

对于pandas中的df,
loc：通过行、列的名称或标签来索引
iloc：通过行、列的索引位置来寻找数据
在下述文件中给出了一个访问癌症患者数据的程序,实现了逻辑回归

 [12_breast_cancer_wisconsin_original.py](12_breast_cancer_wisconsin_original.py) 

##### 混淆矩阵

**TP（True Positive）**：真正例，实际为正例且被预测为正例的样本数。

**TN（True Negative）**：真负例，实际为负例且被预测为负例的样本数。

**FP（False Positive）**：假正例，实际为负例但被预测为正例的样本数。

**FN（False Negative）**：假负例，实际为正例但被预测为负例的样本数。
$$
1\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\;预测结果
\\真实结果
\begin{bmatrix} 
\text{TP} & \text{FP} \\
\text{FN} & \text{TN}
\end{bmatrix}
$$

准确度$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}
$$正确率

精确率$$
\text{Precision} = \frac{TP}{TP + FP}
$$**真正例+伪正例**

召回率$$
\text{Recall} = \frac{TP}{TP + FN}
$$**真正例+伪反例**

$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}= \frac{2TP}{2TP+FN+FP}$

```python
sklearn.metrics.classification_report(y_true,y_pred,labels=[真实值标签],target_names=None(替换混淆矩阵的标签,用元组装载))
```

 [12_breast_cancer_wisconsin_original.py](12_breast_cancer_wisconsin_original.py) 代码

##### 分类评估

对于数学天赋,我们笼统认为所有人都没有,正确率高达99.99%
为了解决这种样本不均衡(通常<1:4时)的问题,
我们可以通过ROC曲线和AUC指标来解决

##### ROC曲线和AUC指标

$$
\text{TPR}=\text{Recall} = \frac{TP}{TP + FN}
所有真实类别为1的样本中，预测类别为1的比例\\
\text{FPR}=\frac{FP}{FP + TN}所有真实类别为0的样本中，预测类别为1的比例
$$

![image-20250213003812747](README.assets/image-20250213003812747.png)

ROC 曲线以FPR作为横轴，TPR作为纵轴，展示模型在不同阈值下的表现。随着分类阈值的变化，模型的 FPR 和 TPR 也会相应变化，绘制出的曲线即为 ROC 曲线。

理想情况下,ROC 曲线应尽可能接近左上角，表示模型在高 TPR 的同时保持低 FPR

AUC（Area Under the Curve，曲线下面积）是 ROC 曲线下方的面积，数值范围从 0 到 1。**AUC 值越接近 1，表示模型的分类性能越好**；AUC 值为 0.5 时，表示模型的分类能力与随机猜测相当(等于乱猜)。AUC 提供了一个综合的评估指标，能够有效地比较不同模型的性能。
**AUC 的直观解释是：在所有可能的分类阈值下，模型将正样本的预测分数排在负样本之前的概率。**

缺点:仅适用于二分类的分类器



##### 处理样本不均衡问题

``pip3 install imbalanced-learn``下载该包

   [13_supplement_imblearn.py](13_supplement_imblearn.py) 代码如链接

###### 过采样方法

​	增加数国少那一类样本的数量

###### 欠采样方法

​		减少数量较多那一类样本的数量

**随机过采样**
在少数类中随机选择⼀些样本，然后通过复制所选择的样本生成样本集。

缺点:训练复杂度增加,容易过拟合
优化:使用SMOTE算法(合成少数类过采样技术)

##### **smote**:

在点间做平滑而不是直接重复样本点,

![image-20250213170313172](README.assets/image-20250213170313172.png)

<img src="README.assets/image-20250213170545512.png" alt="image-20250213170545512" style="zoom:50%;" /><img src="README.assets/image-20250213165543191.png" alt="image-20250213165543191" style="zoom:50%;" />



##### 随机欠采样:

可能丢失重要特征

![image-20250213170313172](README.assets/image-20250213170313172.png)

<img src="README.assets/image-20250213170656835.png" alt="image-20250213170656835" style="zoom:50%;" />





### 决策树

常见的**ID3,C4.5,CART**算法

#### ID3

使用信息熵作为划分属性

###### 信息熵

公式如下
$$
H(X) = - \sum_{i=1}^{n} p(x_i) \log p(x_i)
$$
用不确定性(熵)的减少判断信息含量

例子1：假如有三个类别，分别占比为：{1/3,1/3,1/3}，信息熵计算结果为：

$H=-\frac{1}{3}\log(\frac{1}{3})-\frac{1}{3}\log(\frac{1}{3})-\frac{1}{3}\log(\frac{1}{3})=1.0986$

例子2：假如有三个类别，分别占比为：{1/10,2/10,7/10}，信息熵计算结果为：

$H=-\frac{1}{10}\log(\frac{1}{10})-\frac{2}{10}\log(\frac{2}{10})-\frac{7}{10}\log(\frac{7}{10})=0.8018$

**熵越大，表示整个系统不确定性越大，越随机，反之确定性越强。**

特征$A$对训练数据集D的信息增益$g(D,A)$，定义为集合$D$的熵$H(D)$与特征A给定条件下D的熵$H(D|A)$之差。(划分前-划分后)

$g(D,A)=H(D)-H(D|A)$

其中

$H(D\mid A)=\sum\limits{i=1}^{n}\frac{\mid D_i\mid}{\mid D\mid}H(D_i)=-\sum\limits{i=1}^{n}\frac{\mid D_i\mid}{\mid D\mid}\sum\limits{k=1}^{K}\frac{\mid D{ik}\mid}{\mid D_i\mid}\log_2\frac{\mid D_{ik}\mid}{\mid D_i\mid}$

一般而言，信息增益越大，则意味着**使用属性 a 来进行划分所获得的"纯度提升"越大**。因此，我们可**用信息增益来进行决策树的划分属性选择**， **ID3 决策树**以信息增益为准则来选择划分属性。 

###### 缺点:

信息增益对于可取数目较多的属性有所偏好(比如用序号作为划分属性),因此**C4.5决策树**使用**信息增益率**来选择**最优划分属性**

#### C4.5

使用信息增益率来划分**多叉树**,解决了上述ID3的缺点

##### 信息增益率 (IGR)

信息增益 (IG) 的公式： 
$$
IG(A) = H(D) - H(D \mid A)
$$

其中，
- \( H(D) \) 表示数据集 \( D \) 的熵，
- \( H(D \mid A) \) 表示在特征 \( A \) 条件下的条件熵。

---

**固有值 (IV) 的公式：**
$$
IV(A) = -\sum_{i=1}^{n} \frac{|D_i|}{|D|} \log_2 \left(\frac{|D_i|}{|D|}\right)
$$

其中，
- \( |D_i| \) 是特征 \( A \) 第 \( i \) 个取值对应的数据子集的大小，
- \( |D| \) 是整个数据集的大小。

---

**信息增益率 (IGR) 的公式：**
$$
IGR(A) = \frac{IG(A)}{IV(A)}
$$

#### CART算法

通过比较基尼值划分**二叉树**,基尼值越小越好

###### 基尼值

数据集中任意两个样本标签不一致的概率
$$
Gini(D) = \sum_{k=1}^{|y|} \sum_{k' \neq k} p_k p_{k'} = 1 - \sum_{k=1}^{|y|} p_k^2
$$
基尼值越小,数据集D纯度越高

###### 基尼指数(经过特征a分割后的基尼值

$$
Gini\_index(D, a) = \sum_{v=1}^{V} \frac{D^v}{D} Gini(D^v)
$$

一般会选划分后基尼系数最小的属性作为最优划分属性

其中：

- Gini_index(D,a) 是特征 a 的 Gini 指数。
- D^v 是特征a 的取值 v 对应的子集。
- D 是数据集 D 的总大小。
- Gini(D^v) 是子集 D^v的 Gini 指数。
- V 是特征a 的取值个数。

**CART算法通过基尼指数**实现
CART同时进行了分类和回归,因此**可以处理离散和连续属性**(ID3,C4.5只能处理离散属性)



#### 剪枝

目的:抛开噪声和过拟合,**提升泛化性能**

##### 预剪枝

在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点；

##### 后剪枝

生成一棵完整的决策树，然后自底向上地**对非叶结点进行考察**，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。

##### 总结

后剪枝通常比预剪枝决策树保留了更多的分支。
一般情形下，后剪枝的欠拟合风险很小，**泛化性能往往优于预剪枝**。
后剪枝的训练开销比未剪枝决策树和预剪枝决策树都要大得多

##### 多变量决策树(如OC1)

通过一组特征的线性组合来决策,建议搜索看看原理

##### [特征提取定义](#特征提取)

通过`sklearn.feature_extraction.DictVectorizer()`和`sklearn.feature_extraction.text.CountVectorizer()`实现
[代码](14_feature_vectorizer&tfidf.py) 

###### Tf-idf

- 主要思想：如果某个词或短语在一篇文章中**出现的概率高**，并且**在其他文章中很少出现**，则认为此**词或者短语**具有很好的**类别区分**能力，适合用来分类。

- 作用：用以**评估一字词**对于一个文件集或一个语料库中的其中一份文件的**重要程度**。



##### 决策树算法api

```python
class sklearn.tree.DecisionTreeClassifier(criterion='gini',max_depth=None,random_state=None)
```



-  criterion: 特征选择标准
  - "gini"或者"entropy",前者代表基尼系数，后者代表信息增益。一默认"gini",即CART算法。
- min_samples_split : 内部节点再划分所需最小样本数
  - 这个值限制了子树继续划分的条件，如果某节点的样本数少于min_samples_split,,则不会继续再尝试选择最优特征来进行划分默认值2。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。如10万样本，建立决策树时，可以参考min_samples_splita=10。
-  min_samples_leaf
  - 叶子节点最少样本数
  - 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝。默认是1，可以输入最少的样本数的整数，或者最少样本数占样本总数的百分比。如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。10万样本项目参考min_samples._leaf的值为5。
-  max_depth
  - 决策树最大深度
  - 决策树的最大深度，默认可以不输入，如果不输入的话，决策树在建立子树的时候不会限制子树的深度。一般来说，数据少或者特征少的时候可以不管这个值。如果模型样本量多，特征也多的情况下，推荐限制这个最大深度，具体的取值取决于数据的分布。常用的可以取值10-100

###### 案例：预测泰坦尼克号生存概率

[数据集链接](https://www.kaggle.com/c/titanic/overview)



02-1

部分代码和笔记可以在下面找到

# TODO

[链接](https://zsyll.blog.csdn.net/category_10993525_2.html)
[链接](https://blog.csdn.net/2201_75415080?type=blog) 

  

