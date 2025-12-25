import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 通过数据集的源地址读取Boston房价数据集
data_url="http://lib.stat.cmu.edu/datasets/boston"
raw_df=pd.read_csv(data_url,sep=r"\s+",skiprows=22,header=None)

# 数据可能分为多行，需要合并
data=np.hstack([raw_df.values[::2,:],raw_df.values[1::2,:2]])
target=raw_df.values[1::2,2]

# 合并特征和目标变量
complete_data=np.column_stack((data,target))
columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
          'PTRATIO', 'B', 'LSTAT', 'MEDV']

# 创建包含特征和目标变量的DataFrame
boston=pd.DataFrame(complete_data,columns=columns)
print(boston.head())

# describe()方法提供了数据集的统计摘要信息，包括计数、均值、标准差、最小值、25%分位数、中位数（50%分位数）、75%分位数和最大值。
print(boston.describe())

# 绘制直方图
# boston.hist(bins=20,figsize=(20,15))
# plt.show()

# 绘制各个特征与房价的关系
plt.rcParams['font.sans-serif']=['SimHei'] # 使用黑体
plt.rcParams['axes.unicode_minus']=False # 解决符号显示问题

# 计算相关矩阵
correlation_matrix=boston.corr()

# 可视化矩阵
# plt.figure(figsize=(12,10))
# sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt=".2f")
# plt.show()


warnings.filterwarnings("ignore")
# 可视化刚刚选择的特征RM与房价的关系
# plt.scatter(boston['RM'],boston['MEDV'],alpha=0.5)
# plt.xlabel('平均房价数(RM)')
# plt.ylabel('房价(MEDV)')
# plt.title('RM与MEDV的关系')
# plt.show()

# 可视化刚刚选择的特征LSTAT与房价的关系
# plt.scatter(boston['LSTAT'],boston['MEDV'],alpha=0.5)
# plt.xlabel('低收入者比例(LSTAT)')
# plt.ylabel('房价(MEDV)')
# plt.title('LSTAT与MEDV的关系')
# plt.show()

# 异常值处理
# 识别RM中的异常值
plt.boxplot(boston['RM'])
# plt.show()
# 将RM中大于8的异常值替换为8
boston.loc[boston['RM']>8,'RM']=8

# 多元线性回归
# 拆分特征和目标变量
x=boston.drop('MEDV',axis=1)
y=boston['MEDV']

# 数据分割，拆分出模型训练和测试数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

# 创建模型
model=LinearRegression()

# 模型训练，传入训练参数
model.fit(x_train,y_train)

# 模型预测，传入测试数据集
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f"均方误差(MSE):{mse:.2f}")
print(f"决定系数 R^2值:{r2:.2f}") # R²接近1 ：模型拟合效果好

plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel('实际房价(MEDV)')
plt.ylabel('预测房价(MEDV)')
plt.title('实际房价 vs 预测房价')
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],'r--')
plt.show(

)

# 自己填写数据集，测试模型
new_data=np.array([[0.02731,0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,396.9,9.14]])
predicted_price=model.predict(new_data)
print(f"预测房价:{predicted_price[0]:.2f}")
