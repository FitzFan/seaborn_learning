#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

# load data
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

# show skewness and Kurtosis  偏态和峰度
print("Skewness : %f " % df_train['SalePrice'].skew())
print("Kurtosis : %f " % df_train['SalePrice'].kurt())


# 设置画布大小为A4值
a4_dims = (11.7, 8.27) # A4纸大小


# 单变量的分布
with sns.axes_style("darkgrid"): # 比较喜欢的主题
    f, ax = plt.subplots(figsize=a4_dims)
    fig = sns.distplot(df_train['SalePrice'], color='orange')
    fig.set_title('Flexibly plot a univariate distribution of observations.', weight='bold') # 设置标题
plt.show()


# 多变量散点图，也可以理解为 回归图
with sns.axes_style("darkgrid"):
    f, ax = plt.subplots(figsize=a4_dims)
    data = pd.concat([df_train['SalePrice'],df_train["GrLivArea"]],axis = 1)
    fig = sns.regplot(x="GrLivArea", y="SalePrice", data=data, fit_reg=False) # fit_reg is True, estimate and plot a regression model relating the x and y variables.
    fig.axis(xmin=0, xmax=6000, ymin=0,ymax=800000)
    fig.set_title('Plot data and a linear regression model fit.', weight='bold') # 设置标题
plt.show()


# box plot overallqual / saleprice  
data = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis = 1)
with sns.axes_style("darkgrid"):
    f, ax = plt.subplots(figsize=a4_dims)
    fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data, ax=ax)
    fig.axis(ymin=0, ymax=800000)
    fig.set_title('Draw a box plot to show distributions with respect to categories.', weight='bold').set_fontsize('14') # 给标题设置字体和加粗
plt.show()


# boxplot saleprice / yearbuilt  
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train['YearBuilt']],axis = 1)
with sns.axes_style("darkgrid"):
    f, ax = plt.subplots(figsize=a4_dims)
    fig = sns.boxplot(x=var, y='SalePrice', data=data, ax=ax) 
    fig.axis(ymin=0,ymax=800000)  
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90) # x轴标签，旋转90度
    fig.set_title('Draw a box plot to show distributions with respect to categories.', weight='bold') # 给标题设置字体和加粗
plt.show()


# correlation matrix, 相关矩阵  
corrmat = df_train.corr() 
f, ax = plt.subplots(figsize=a4_dims) 
# make limits of the colormap is between -1 and 1 and plot a heatmap for data centered on 0 with a diverging colormap
fig = sns.heatmap(corrmat, cmap='RdBu', linewidths=0.05, vmin=-1, vmax=1, center=0) 
fig.set_title('Plot rectangular data as a color-encoded matrix.', weight='bold') # 设置标题
plt.show()

# 取与目标变量相关系数最大的10个和最小的10个变量，绘制重要变量的相关系数热力图
top_num = 10  
cols_max = corrmat.nlargest(top_num,'SalePrice')['SalePrice'].index   # 取出与saleprice相关性最大的十项
cols_min = corrmat.nsmallest(top_num,'SalePrice')['SalePrice'].index  # 取出与saleprice相关性最小的十项
cols = list(cols_max) + list(cols_min)
cols = list(set(cols))
cm = np.corrcoef(df_train[cols].values.T)  #相关系数
f, ax = plt.subplots(figsize=a4_dims) 
hm_first = sns.heatmap(cm, linewidths=0.05, cbar=True, vmin=-1, vmax=1, center=0,
                      annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols, xticklabels=cols, cmap='RdBu')

# 突出一些特殊的数字
for text in hm_first.texts:
    value_num = abs(float(text.get_text())) # 取绝对值
    if value_num >= 0.7 and value_num != 1.00:
        text.set_fontsize(9)
        text.set_weight('bold')
        text.set_color('orangered')
hm_first.set_title('Plot rectangular data as a color-encoded matrix of important variables', weight='bold') # 设置标题
plt.show()

"""
热力图参数说明：
- annotate的缩写，annot默认为False，当annot为True时，在heatmap中每个方格写入数据，annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等
- fmt: String formatting code to use when adding annotations.
- linewidths: Width of the lines that will divide each cell.
- cbar: Whether to draw a colorbar.
- square: If True, set the Axes aspect to “equal” so each cell will be square-shaped.
- annot_kws: Keyword arguments for ax.text when annot is True.
- linewidths: Width of the lines that will divide each cell. 相邻单元格之间的距离
"""

# 查看各字段的缺失，以百分比的形式显示
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])
missing_data.head(20)


# 数据标准化的demo，此处以“SalePrice”为例。
with sns.axes_style("darkgrid"):
    f, ax = plt.subplots(figsize=a4_dims) # 设置画布大小
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
    high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
    fig = sns.distplot(saleprice_scaled)
    fig.set_title('Flexibly plot a univariate distribution of observations.', weight='bold') # 设置标题
plt.show()

print('outer range(low) of the distribution :','\n',low_range)
print ('outer range (high) of the distribution :','\n',high_range)

# 双变量数据分析  Bivariate analysis
var = 'GrLivArea'  
data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)
with sns.axes_style("darkgrid"):
    # sns.jointplot(x=var, y='SalePrice',data=data,kind='kde')
    # sns.jointplot(x=var, y='SalePrice',data=data,kind='scatter')
    fig = sns.jointplot(x=var, y='SalePrice', data=data, kind='reg')  # this kind may be better.
    # sns.jointplot(x=var, y='SalePrice',data=data,kind='resid')
    # sns.jointplot(x=var, y='SalePrice',data=data,kind='hex')
plt.show()


# 正态性检验
# in the search for normality  
# histogram and normal probability plot  直方图和正态概率图
# 设置画布_1的size
with sns.axes_style("darkgrid"):
    f, ax= plt.subplots(figsize=a4_dims)
    fig = sns.distplot(df_train['SalePrice'], fit = norm, ax=ax) # fit 控制拟合的参数分布图形
    fig.set_title('Flexibly plot a univariate distribution of observations.', weight='bold') # 设置标题
plt.show()

with sns.axes_style("darkgrid"):
    # 设置画布_2的size
    f, ax= plt.subplots(figsize=a4_dims)
    # probplot :Calculate quantiles for a probability plot, and optionally show the plot. 计算概率图的分位数
    res = stats.probplot(df_train['SalePrice'], plot=plt) # seaborn画不了Q-Q图~
plt.show()


# 一次性画多个图的demo
# 将画布设置为一列两行
with sns.axes_style("darkgrid"):
    f, (ax_1, ax_2)= plt.subplots(ncols=1, nrows=2, figsize=a4_dims)
    # 第一行的图形
    fig = sns.distplot(df_train['SalePrice'], fit = norm, ax=ax_1) # fit 控制拟合的参数分布图形
    fig.set_title('Flexibly plot a univariate distribution of observations In subplots', weight='bold') # 设置标题
    # 第二行的图形
    sns.distplot(saleprice_scaled, ax=ax_2)

# Finalize the plot
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=2) # 有多少张图, h_pad为多少
plt.show()


# refer:
# https://blog.csdn.net/Amy_mm/article/details/79538083
# https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python


"""
学习记录：
1、无法保存图片。
2、学习更多的画图方法。

"""