---
title: 贝叶斯公式/朴素贝叶斯分类器及python自实现
date: 2018-11-20 09:48:40
tags: Machine Learning
category: Machine Learning
mathjax: true
---



本文从贝叶斯与频率概率的对比入手理解贝叶斯决策的思维方式。通过两个实例理解贝叶斯的思想与流程，然后梳理了朴素贝叶斯分类器的算法流程，最后从零开始实现了朴素分类器的算法。

### 1.起源、提出与贝叶斯公式

贝叶斯派别主要是与古典统计学相比较区分的。

古典统计学基于大数定理，将一件事件经过大量重复实验得到的频率作为事件发生的概率，如常说的掷一枚硬币，正面朝上的概率为0.5。但，如果事先知道硬币的密度分布并不均匀，那此时的概率是不是就不为0.5呢？这种不同于“非黑即白”的思考方式，就是贝叶斯式的思考方式。

贝叶斯除了提出上述思考方式之外，还特别提出了举世闻名的贝叶斯定理。贝叶斯公式：

$$ P(B_i|A) = \frac{P(B_i)P(A|B_i)}{\sum_{j=1}^nP(B_j)P(A|B_j)}$$

这里通过全概率公式，我们可以将上式等同于

$$ P(B|A) = \frac{P(B)P(A|B)}{P(A)} = P(B)\frac{P(A|B)}{P(A)}$$

右边的分式中，分子的P(A)称为先验概率，是B事件未发生前，对A事件概率的判断；P(A|B)即是在B事件发生之后，对A事件发生的后验概率。这整个分式我们也称之为‘’可能性函数(Likelyhood)‘’，这是一个调整因子，对A事件未发生之前B的概率进行调整，以得到A事件发生的前提下B事件发生的后验概率P(B|A)。

以一句话与一张图概括：贝叶斯定理是一种在已知其他概率的情况下求概率的方法。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181120212448460.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzNTkwOTIx,size_16,color_FFFFFF,t_70)


### 2.以实例感受贝叶斯决策：癌症病人计算 问题

> 有一家医院为了研究癌症的诊断，对一大批人作了一次普查，给每人打了试验针，然后进行统计，得到如下统计数字：
>
> 1. 这批人中，每1000人有5个癌症病人；
> 2. 这批人中，每100个正常人有1人对试验的反应为阳性
> 3. 这批人中，每100个癌症病人有95入对试验的反应为阳性。
>
> 通过普查统计，该医院可开展癌症诊断。现在某人试验结果为阳性，诊断结果是什么?

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181120212247159.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzNTkwOTIx,size_16,color_FFFFFF,t_70)
(全是公式实在不好打，就用自己笔记代替啦)

上述例子是基于**最小错误率贝叶斯决策**，即哪类的概率大判断为哪类，也是我们常见的分类决策方法。但是将正常人错判为癌症病人，与将癌症病人错判为正常人的损失是不一样的。将正常人错判为癌症病人，会给他带来短期的精神负担，造成一定的损失，这个损失比较小。如果把癌症病人错判为正常人，致使患者失去挽救的机会，这个损失就大了。这两种不同的错判所造成损失的程度是有显著差别的。

所以，在决策时还要考虑到各种错判所造成的不同损失，由此提出了**最小风险贝叶斯决策**。

我们将$I_{ij}$记为将j类误判为i类所造成的损失。此处类别为2，如将样本x判为癌症病人$c_1$造成损失的数学期望为：

$R_1=I_{11}P(C_1|X)+I_{12}P(C_2|X)$

同理，将样本x判为癌症病人$c_2$造成损失的数学期望为

$R_2=I_{21}P(C_1|X)+I_{22}P(C_2|X)$

选择最小风险作为决策准则，若$R_1<R_2$，则样本X$\epsilon R_1$，否则X$\epsilon R_2$

### 3.以实例感受贝叶斯修正先验概率：狼来了

> 给出一些合理解释：
>
> 事件A表示:“小孩说谎”；事件B表示:“小孩可信”。
>
> 村民起初对这个小孩的信任度为0.8,即P(B)=0.8。
>
> 我们认为可信的孩子说谎的可能性为0.1。即P(A|B)=0.1。
>
> 不可信的孩子说谎的可能性为0.5,即P(A|^B)=0.5(用 ^B表示B的对立事件)。
>
> 求小孩第一次说谎、第二次说谎后，村民对这个小孩的信任度从P(B)=0.8会发生什么变化?

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181120212414543.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzNTkwOTIx,size_16,color_FFFFFF,t_70)

由此我们可以得到贝叶斯决策另一个道理，**贝叶斯决策会不断用后验概率，逐渐修正先验概率。**投射到现实生活，我们也可以理解为当对某件事进行判断决策时，关于此事得到的信息越多，对此事的决策越准，而绝非yes/no的五五开决策。

> 还是狼来的例子，如果这个孩子要改邪归正，他需要多少次才能把信任度提高到80%？

如果要把信任度提高，那接下来就需要通过几次 不说谎 的经历来修正村民的观念，那可信度的计算，要记得将P(A|B)换成P((fei)A|B)，(A上方的横线打不出来...)，就是可信的孩子不说谎的可能性为1-P(A|B)=1-0.1，同样，不可信的孩子不说谎的概率为1-0.5

此时，我们就要用代码来实现而不是手算啦。代码如下：

```python
def calculateTrustDegree(pb):
  PB_A = float((1-PA_B)*pb)/((1-PA_B)*pb+(1-PA_Bf)*(1-pb))
  return PB_A
PA_B = 0.1
PA_Bf = 0.5
pb = 0.138
N = 0
while(pb <=0.8):
    pb = calculateTrustDegree(pb)
    N+=1
print("He need {0} times with honest saying,villager can trust him again.".format(N))

# N = 6
# He need 6 times with honest saying,villager can trust him again.
```
### 4.朴素贝叶斯分类器

**贝叶斯分类算法**是一大类分类算法的总称。

贝叶斯分类算法以**样本可能属于某类的概率来作为分类依据**。

**朴素贝叶斯分类算法**是贝叶斯分类算法中最简单的一种，采用了**“属性条件独立性假设”**：对已知类别，假设所有属性互相独立，也就是在概率的计算时，可以将一个类别的每个属性直接相乘。这在概率论中应该学过，两个事件独立时，两个事件同时发生的概率等于两个事件分别发生的乘积。

给定一个属性值，其属于某个类的概率叫做条件概率。对于一个给定的类值，将每个属性的条件概率相乘，便得到一个数据样本属于某个类的概率。

我们可以通过计算样本归属于每个类的概率，然后选择具有最高概率的类来做预测。

我们以鸢尾花分类实例来过算法流程，并不适用sklearn库自实现朴素贝叶斯分类器。

> 实例数据集介绍：鸢尾花数据集包含4维的特征，共三类150个样本，每类均有50个样本。

算法流程概括如下：

1. 计算样本属于某类的**先验概率**，属于A类的概率为$\frac{属于A类的样本数}{所有样本数}$，以此类推
2. 计算**类条件概率**，离散值通过类别数量比值，此数据集的属性特征为连续值所以通过 **概率密度函数** 来计算。首先计算在一个属性的前提下，该样本属于某类的概率；相乘合并所有属性的概率，即为某个数据样本属于某类的类条件概率 
   - 计算每个特征属于每类的条件概率
     - 概率密度函数实现
     - 计算每个属性的均值和方差
     - 按类别提取属性特征，这里会得到 类别数目*属性数目 组 （均值，方差）
   - 按类别将每个属性的条件概率相乘，如下所示
     - 判断为A类的概率：p(A|特征1)*p(A|特征2)*p(A|特征3)*p(A|特征4).....
     - 判断为B类的概率：p(B|特征1)*p(B|特征2)*p(B|特征3)*p(B|特征4).....
     - 判断为C类的概率：p(C|特征1)*p(C|特征2)*p(C|特征3)*p(C|特征4).....
3. **先验概率*类条件概率**，回顾一下贝叶斯公式，$$ P(B_i|A) = \frac{P(B_i)P(A|B_i)}{\sum_{j=1}^nP(B_j)P(A|B_j)}$$。由于样本确定时，贝叶斯公式的分母都是相同的。所以判断样本属于哪类只需要比较分子部分：先验概率*类条件概率，最终属于哪类的概率最大，则判别为哪类，此处为**最小错误率贝叶斯分类**，若采用最小风险需要加上判断为每个类别的风险损失值。

### 5.代码实现

#### 1.数据集载入，划分训练集与测试集

```python
data_df = pd.read_csv('IrisData.csv')
def splitData(data_list,ratio):
  train_size = int(len(data_list)*ratio)
  random.shuffle(data_list)
  train_set = data_list[:train_size]
  test_set = data_list[train_size:]
  return train_set,test_set

data_list = np.array(data_df).tolist()
trainset,testset = splitData(data_list,ratio = 0.7)
print('Split {0} samples into {1} train and {2} test samples '.format(len(data_df), len(trainset), len(testset)))

# Split 150 samples into 105 train and 45 test samples 
```
#### 2.计算先验概率

此时需要先知道数据集中属于各类别的样本分别有多少。我们通过一个函数实现按类别划分数据。

两个返回值分别为划分好的数据字典，以及划分好的数据集中每个类别的样本数

```python
def seprateByClass(dataset):
  seprate_dict = {}
  info_dict = {}
  for vector in dataset:
      if vector[-1] not in seprate_dict:
          seprate_dict[vector[-1]] = []
          info_dict[vector[-1]] = 0
      seprate_dict[vector[-1]].append(vector)
      info_dict[vector[-1]] +=1
  return seprate_dict,info_dict

train_separated,train_info = seprateByClass(trainset)

# train_info：
# {'Setosa': 41, 'Versicolour': 33, 'Virginica': 31}
```
计算属于每个类别的先验概率

```python
def calulateClassPriorProb(dataset,dataset_info):
  dataset_prior_prob = {}
  sample_sum = len(dataset)
  for class_value, sample_nums in dataset_info.items():
      dataset_prior_prob[class_value] = sample_nums/float(sample_sum)
  return dataset_prior_prob

prior_prob = calulateClassPriorProb(trainset,train_info)

#{'Setosa': 0.3904761904761905,
# 'Versicolour': 0.3142857142857143,
# 'Virginica': 0.29523809523809524}
```
#### 3.计算类条件概率

3.1 首先计算每个特征属于每类的条件概率，前面说过这里我们使用概率密度函数来计算

概率密度函数实现：

方差公式：$ var = \frac{\sum(x-avg)^{2}}{n-1}$，概率密度函数：$ p(xi|c) = \frac{1}{\sqrt{2\pi}\sigma_{c,i}}exp(-\frac{(xi-mean_{c,i})^{2}}{2\sigma_{c,i}^{2}})$ , $\sigma$是标准差（方差开方）

```python
# 均值
def mean(list):
  list = [float(x) for x in list] #字符串转数字
  return sum(list)/float(len(list))
# 方差
def var(list):
  list = [float(x) for x in list]
  avg = mean(list)
  var = sum([math.pow((x-avg),2) for x in list])/float(len(list)-1)
  return var
# 概率密度函数
def calculateProb(x,mean,var):
    exponent = math.exp(math.pow((x-mean),2)/(-2*var))
    p = (1/math.sqrt(2*math.pi*var))*exponent
    return p
```
每个属性特征属于每类的条件概率是个组合。举例来说，这里有3个类和4个数值属性，然后我们需要每一个属性（4）和类（3）的组合的类条件概率。

为了得到这12个概率密度函数，那我们需要提前知道这12个属性分别的均值和方差，才可以带入到上述概率密度函数中计算。

计算每个属性的均值和方差：

```python
def summarizeAttribute(dataset):
    dataset = np.delete(dataset,-1,axis = 1) # delete label
    summaries = [(mean(attr),var(attr)) for attr in zip(*dataset)]
    return summaries

summary = summarizeAttribute(trainset)
#[(5.758095238095239, 0.7345732600732595),
# (3.065714285714285, 0.18592857142857133),
# (3.5533333333333323, 3.2627051282051274),
# (1.1142857142857148, 0.6014285714285714)]
```
按类别提取属性特征，这里会得到 类别数目*属性数目 组 （均值，方差）

```python
def summarizeByClass(dataset):
  dataset_separated,dataset_info = seprateByClass(dataset)
  summarize_by_class = {}
  for classValue, vector in dataset_separated.items():
      summarize_by_class[classValue] = summarizeAttribute(vector)
  return summarize_by_class

train_Summary_by_class = summarizeByClass(trainset)
#{'Setosa': [(4.982926829268291, 0.12445121951219511),
#  (3.3975609756097565, 0.1417439024390244),
#  (1.4707317073170731, 0.03412195121951221),
#  (0.24390243902439032, 0.012024390243902434)],
# 'Versicolour': [(5.933333333333334, 0.2766666666666667),
#  (2.7909090909090906, 0.08960227272727272),
#  (4.254545454545454, 0.23755681818181815),
#  (1.33030303030303, 0.03905303030303031)],
# 'Virginica': [(6.596774193548387, 0.5036559139784946),
#  (2.9193548387096775, 0.10427956989247314),
#  (5.5612903225806445, 0.37711827956989247),
#  (2.0354838709677416, 0.06369892473118278)]}

```
按类别将每个属性的条件概率相乘。

我们前面已经将训练数据集按类别分好，这里就可以实现，输入的测试数据依据每类的每个属性（就那个类别数*属性数的字典）计算属于某类的类条件概率。

```python
def calculateClassProb(input_data,train_Summary_by_class):
  prob = {}
  for class_value, summary in train_Summary_by_class.items():
      prob[class_value] = 1
      for i in range(len(summary)):
          mean,var = summary[i]
          x = input_data[i]
          p = calculateProb(x,mean,var)
      prob[class_value] *=p
  return prob

input_vector = testset[1]
input_data = input_vector[:-1]
train_Summary_by_class = summarizeByClass(trainset)
class_prob = calculateClassProb(input_data,train_Summary_by_class)

#{'Setosa': 3.3579279836005993,
# 'Versicolour': 1.5896628317396685e-07,
# 'Virginica': 5.176617264913899e-12}
```
#### 4.先验概率*类条件概率

朴素贝叶斯分类器

```python
def bayesianPredictOneSample(input_data):
  prior_prob = calulateClassPriorProb(trainset,train_info)
  train_Summary_by_class = summarizeByClass(trainset)
  classprob_dict = calculateClassProb(input_data,train_Summary_by_class)
  result = {}
  for class_value,class_prob in classprob_dict.items():
      p = class_prob*prior_prob[class_value]
      result[class_value] = p
  return max(result,key=result.get)
```
终于把分类器写完啦，接下来就让我们看看测试数据的结果！

单个样本测试：

```python
input_vector = testset[1]
input_data = input_vector[:-1]
result = bayesianPredictOneSample(input_data)
print("the sameple is predicted to class: {0}.".format(result))

# the sameple is predicted to class: Versicolour.
```
看看分类准确率

```python
def calculateAccByBeyesian(dataset):
  correct = 0
  for vector in dataset:
      input_data = vector[:-1]
      label = vector[-1]
      result = bayesianPredictOneSample(input_data)
      if result == label:
          correct+=1
  return correct/len(dataset)

acc = calculateAccByBeyesian(testset)
# 0.9333333333333333
```
全部代码及数据集已上传至[github](https://github.com/wenweny/Machine-Learning/tree/master/bayesian)，第一次写博客不足之处欢迎大家提出交流学习。



### 6. 参考文献

[机器学习之用Python从零实现贝叶斯分类器](http://python.jobbole.com/81019/)

[知乎-你对贝叶斯统计都有怎样的理解？](https://www.zhihu.com/question/21134457/answer/169523403)

[贝叶斯公式由浅入深大讲解—AI基础算法入门](https://www.cnblogs.com/zhoulujun/p/8893393.html)

[先验乘后验贝叶斯定理](http://dy.163.com/v2/article/detail/CU0MJOCV05118CTM.html)