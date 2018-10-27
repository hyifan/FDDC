### 赛题：
A股上市公司季度营收预测

### 链接：
https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11409106.5678.1.4ee017ffKtllcD&raceId=231660

### 特征工程思路：
1、去除缺省值：统计财务报表中缺失值情况，去除缺失值大于70%的字段<br/>
2、映射宏观数据与微观数据：利用仅有的金融知识形成map表，将宏观数据与微观数据信息映射到各只股票<br/>
3、将全部特征归一化：(数据 - 均值) / 标准差<br/>
4、形成训练集：该赛题预测的是2018年S1季度的营业收入，训练集将各年的S1季度营业收入作为标记信息，以往8个季度的字段作为特征，形成一条数据。如：将2017S1的营业收入作为标记信息，则“2015S1、2015Q3、2015A、2016Q1、2016S1、2016Q3、2016A、2017Q1”的各个字段结合为特征。<br/>
5、形成用于最后预测的数据集：用相同顺序将“2016S1、2016Q3、2016A、2017Q1、2017S1、2017Q3、2017A”的各个信息结合作为特征，用于最后预测2018年S1季度的营业收入。

### 算法思路：
1、采用前馈神经网络的思想，选取反向传播算法快速下降代价函数的梯度，计算权重weights和偏置biases<br/>
2、选用的代价函数为（label为标记信息，a为预测值，market为市值）：C = (1/2) * sum((label - a) ** 2 * log(max(market,2))<br/>
3、选用的激活函数为：a = ln (1 + e ** x)<br/>
4、代码执行过程中，会在各个迭代期结束后，输出测试集的以大赛标准执行的评估指标值，通过观察评估指标下降趋势选取最优的参数。

### 在终端运行的指令：
``` python
import loader
import network

# 获取训练集（training_data）与用于最后预测的数据集(answer_data)
# _bs为一般工商业，_fa为金融行业
# symbol为answer_data对应的股票代码，symbol与answer_data按顺序一一对应
training_data_bs, training_data_fa, answer_data_bs, answer_data_fa, symbol_bs, symbol_fa = loader.load_training_data()

# 传入各参数，调试得到最终的weights, biases
# training_data       训练集
# test_data           测试集
# epochs              迭代次数
# mini_batch_size     小批量数据的大小
# eta                 速率
# lmbda               规范化参数，防止过拟合
# length              神经网络第一层神经元数量，即特征数量
# num1, num2, ...     神经网络第二、三...层神经元数量
weights, biases = network.main(training_data, test_data, epochs, mini_batch_size, eta, lmbda, [length, num1, num2, ... , 1])

# 该函数用于获取最终结果
# answer_data为用于最后预测的数据集，symbol是与之对应的股票代码
# weights, biases为模型训练得出的值
# '文件名'为最终导出的json表的名称，json表将以股票代码作为key，预测的营业收入为value
network.get_result(answer_data, symbol, weights, biases,'文件名')
```
