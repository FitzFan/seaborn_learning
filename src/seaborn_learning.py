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


# with sns.axes_style("darkgrid"): # 比较喜欢的主题
    # f, ax = plt.subplots(figsize=a4_dims)
    # fig = sns.countplot(x='Street',  data=df_train) # 离散型变量取值分布
    # fig = sns.distplot(a=df_train[df_train['LotFrontage'].notnull()]['LotFrontage']) # 连续性变量取值分布
    # fig = sns.countplot(x='LandContour', hue='LotShape', data=df_train, palette="Set3") # 离散型变量+分类变量取值分布

    # 离散型变量+连续性变量+分类变量分布
    # fig = sns.barplot(x='LandContour',y='LotConfig',data=df_train, hue='LotShape',palette="Set3",estimator=np.median) 
#     fig.set_title('Value Distribution Of %s'%('LandContour'), weight='bold') # 设置标题
#     # fig = sns.countplot(x="class", hue='who',data=titanic)
# plt.show()
# sys.exit()

# 分面图
# titanic = sns.load_dataset("titanic")
# with sns.axes_style("darkgrid"): 
    # strip: 散点图, 相同取值的点会重合。
    # swarm: 分散点图，相同取值的点不会重合。
    # bar:条形图，条形是均值，线条的估计：线条的最上面的位置代表最大值的位置
    # box: 箱线图 ----常用
    # count: 频次, 用这个kind，不需要传 y 参数 ---- 常用
    # violin: 小提琴图
    # fig = sns.stripplot(x="LandContour", y="LotFrontage",data=df_train)
    # fig = sns.swarmplot(x="LandContour", y="LotFrontage",data=df_train)
    # fig = sns.catplot(x="LandContour", y="LotFrontage", col="LotShape", col_wrap=4, data=df_train, kind="box", size=2.5, aspect=.8, palette="Set3")
    # fig = sns.catplot(x="LandContour", y="LotFrontage", col="LotShape", hue='CentralAir', col_wrap=4, data=df_train, kind="box", size=2.5, aspect=.8)
    
# plt.show()

# sys.exit()


# 回归图
# tips = sns.load_dataset("tips")
# with sns.axes_style("darkgrid"): 
#     fig = sns.lmplot(x="size", y="total_bill", hue="day", data=tips, aspect=.4, x_jitter=.1) # 绘制在一张画布
#     fig = sns.lmplot(x="size", y="total_bill", col="day", data=tips, aspect=.4, x_jitter=.1) # 多张画布，但颜色一样
#     fig = sns.lmplot(x="size", y="total_bill", hue="day", col='day', data=tips, aspect=.4, x_jitter=.1) # 多张画布，颜色不一样

# plt.show()

# sys.exit()

# 双变量密度图
# iris = sns.load_dataset("iris")
# setosa = iris.loc[iris.species == "setosa"]  # 组1
# virginica = iris.loc[iris.species == "virginica"]  # 组2

# with sns.axes_style("darkgrid"): 
#     f, ax = plt.subplots(figsize=a4_dims)
# with sns.axes_style("darkgrid"): 
#     f, ax = plt.subplots(figsize=a4_dims)
#     ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length, cmap="Oranges", shade=True, shade_lowest=False)
#     ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length, cmap="Blues", shade=True, shade_lowest=False)

#     # ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length, cmap="Blues", shade=True, shade_lowest=False)

# plt.show()
# sys.exit()

# 双变量关系图
# tips = sns.load_dataset("tips")
# with sns.axes_style("darkgrid"): 
#     ax = sns.jointplot("total_bill", "tip", data=tips, kind="reg", stat_func=stats.pearsonr)

# plt.show()
# sys.exit()



# 将画布设置为A4纸
# a4_dims = (11.7, 8.27)
# iris = sns.load_dataset("iris")
# with sns.axes_style("darkgrid"):
#     # 设置画布大小
#     # f, ax = plt.subplots(figsize=a4_dims)
#     fig = sns.pairplot(df_train[['BsmtFinType2','BsmtUnfSF','BsmtFinSF2']], hue='BsmtFinType2', palette="Set3")
    
# plt.show()
# sys.exit()

# 多变量散点图，也可以理解为 回归图
# with sns.axes_style("darkgrid"):
#     f, ax = plt.subplots(figsize=a4_dims)
#     data = pd.concat([df_train['SalePrice'],df_train["GrLivArea"]],axis = 1)
#     fig = sns.regplot(x="GrLivArea", y="SalePrice", data=data, fit_reg=True) # fit_reg is True, estimate and plot a regression model relating the x and y variables.
#     fig.axis(xmin=0, xmax=6000, ymin=0,ymax=800000)
#     fig.set_title('Plot data and a linear regression model fit.', weight='bold') # 设置标题
# plt.show()
# sys.exit()



# box plot overallqual / saleprice  
# data = pd.concat([df_train['SalePrice'],df_train['OverallQual']],axis = 1)
# with sns.axes_style("darkgrid"):
#     f, ax = plt.subplots(figsize=a4_dims)
#     fig = sns.boxplot(x='OverallQual', y='SalePrice', data=data, ax=ax, palette="Set3")
#     # fig.axis(ymin=0, ymax=800000)
#     fig.set_title('Draw a box plot to show distributions with respect to categories.', weight='bold').set_fontsize('14') # 给标题设置字体和加粗
# plt.show()
# sys.exit()

# boxplot saleprice / yearbuilt  
# var = 'YearBuilt'
# data = pd.concat([df_train['SalePrice'],df_train['YearBuilt']],axis = 1)
# with sns.axes_style("darkgrid"):
#     f, ax = plt.subplots(figsize=a4_dims)
#     fig = sns.boxplot(x=var, y='SalePrice', data=data, ax=ax) 
#     fig.axis(ymin=0,ymax=800000)  
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90) # x轴标签，旋转90度
#     fig.set_title('Draw a box plot to show distributions with respect to categories.', weight='bold') # 给标题设置字体和加粗
# plt.show()
# sys.exit()

# correlation matrix, 相关矩阵  
corrmat = df_train.corr() 
# f, ax = plt.subplots(figsize=a4_dims) 
# # make limits of the colormap is between -1 and 1 and plot a heatmap for data centered on 0 with a diverging colormap
# fig = sns.heatmap(corrmat, cmap='RdBu', linewidths=0.05, vmin=-1, vmax=1, center=0) 
# fig.set_title('Plot rectangular data as a color-encoded matrix.', weight='bold') # 设置标题
# plt.show()
# sys.exit()

# 取与目标变量相关系数最大的10个和最小的10个变量，绘制重要变量的相关系数热力图
# top_num = len(df_train)
# cols_max = corrmat.nlargest(top_num,'SalePrice')['SalePrice'].index   # 取出与saleprice相关性最大的十项
# cols_min = corrmat.nsmallest(top_num,'SalePrice')['SalePrice'].index  # 取出与saleprice相关性最小的十项
# cols = list(cols_max) + list(cols_min)
# cols = list(set(cols))

# cols = corrmat.nlargest(top_num, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)  #相关系数
# sns.set(font_scale=1.25)
# hm_first = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
#                          yticklabels=cols.values, xticklabels=cols.values)
# plt.show()
# sys.exit()

# with sns.axes_style("darkgrid"):
#     f, ax = plt.subplots(figsize=a4_dims) 
#     # hm_first = sns.heatmap(cm, linewidths=0.1, cbar=True, vmin=-1, vmax=1, center=0,
#     #                    annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols, xticklabels=cols, cmap='RdBu')
#     hm_first = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
#                          yticklabels=cols, xticklabels=cols)



# # 只突出一些特殊的数字
# for text in hm_first.texts:
#     value_num = abs(float(text.get_text())) # 取绝对值
#     if value_num >= 0.6 and value_num != 1.00:
#         text.set_fontsize(8)
#         text.set_weight('bold')
#         text.set_color('orangered')
#     else:
#         text.set_alpha(0)
# hm_first.set_title('Plot rectangular data as a color-encoded matrix of important variables', weight='bold') # 设置标题
# plt.show()
# sys.exit()
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
# total = df_train.isnull().sum().sort_values(ascending=False)
# percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending = False)
# missing_data = pd.concat([total,percent], axis=1, keys=['Total','Percent'])
# missing_data.head(20)
# sys.exit()

# 数据标准化的demo，此处以“SalePrice”为例。
# with sns.axes_style("darkgrid"):
#     f, ax = plt.subplots(figsize=a4_dims) # 设置画布大小
#     saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
#     low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
#     high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
#     fig = sns.distplot(saleprice_scaled)
#     fig.set_title('Flexibly plot a univariate distribution of observations.', weight='bold') # 设置标题
# plt.show()

# print('outer range(low) of the distribution :','\n',low_range)
# print ('outer range (high) of the distribution :','\n',high_range)
# sys.exit()


# 双变量数据分析  Bivariate analysis
# var = 'GrLivArea'  
# data = pd.concat([df_train['SalePrice'],df_train[var]], axis=1)
# with sns.axes_style("darkgrid"):
#     # sns.jointplot(x=var, y='SalePrice',data=data,kind='kde')
#     # sns.jointplot(x=var, y='SalePrice',data=data,kind='scatter')
#     fig = sns.jointplot(x=var, y='SalePrice', data=data, kind='reg')  # this kind may be better.
#     # sns.jointplot(x=var, y='SalePrice',data=data,kind='resid')
#     # sns.jointplot(x=var, y='SalePrice',data=data,kind='hex')
# plt.show()


# 正态性检验
# in the search for normality  
# histogram and normal probability plot  直方图和正态概率图
# 设置画布_1的size
# with sns.axes_style("darkgrid"):
#     f, ax= plt.subplots(figsize=a4_dims)
#     fig = sns.distplot(df_train['SalePrice'], fit = norm, ax=ax) # fit 控制拟合的参数分布图形
#     fig.set_title('Flexibly plot a univariate distribution of observations.', weight='bold') # 设置标题
# plt.show()
# sys.exit()


# with sns.axes_style("darkgrid"):
#     # 设置画布_2的size
#     f, ax= plt.subplots(figsize=a4_dims)
#     # probplot :Calculate quantiles for a probability plot, and optionally show the plot. 计算概率图的分位数
#     res = stats.probplot(df_train['SalePrice'], plot=plt) # seaborn画不了Q-Q图~
# plt.show()
# sys.exit()


all_df = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']), axis=0, ignore_index=True)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
qualitative = [f for f in all_df.columns if all_df.dtypes[f] == 'object']
print("quantitative: {}, qualitative: {}" .format (len(quantitative),len(qualitative)))

f = pd.melt(all_df, value_vars=quantitative[0:8])
with sns.axes_style("darkgrid"):
    g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
plt.show()
sys.exit()

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