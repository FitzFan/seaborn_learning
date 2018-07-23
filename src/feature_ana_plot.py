#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

"""
- Show the counts of observations in each categorical bin using bars.
- 单离散型变量的取值分布图，可以加上分类变量
"""
def cate_sig_dis(data_frame, col_name, hue_name=None):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax = plt.subplots(figsize=a4_dims)

		# 判断绘图类型
		if hue_name is None:
			# 绘图
			fig = sns.countplot(x=col_name,  data=data_frame, palette="Set3")
		else:
			# 绘图
			fig = sns.countplot(x=col_name, hue=hue_name, data=data_frame, palette="Set3")

		# 设置标题
		# fig.set_title('Value Distribution Of %s'%(col_name), weight='bold')
		
	plt.show()


"""
- Flexibly plot a univariate distribution of observations.
- 单一连续型变量取值分布图
"""
def cont_sig_dis(data_frame, col_name):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax = plt.subplots(figsize=a4_dims)

		# 绘图
		fig = sns.distplot(a=data_frame[data_frame[col_name].notnull()][col_name], hist=True, rug=True)

	plt.show()

"""
- Fit and plot a univariate or bivariate kernel density estimate.
- 双连续性变量的登高图or密度图
"""
def cont_biv_dis(data_frame, x_val_1, y_val_1, x_val_2=None, y_val_2=None):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	with sns.axes_style("darkgrid"): 
		f, ax = plt.subplots(figsize=a4_dims)
		ax = sns.kdeplot(x_val_1, y_val_1, cmap="Oranges", shade=True, shade_lowest=False)

		# 判断是否需要画多个
		if x_val_2 is not None and y_val_2 is not None:
			ax = sns.kdeplot(x_val_2, y_val_2, cmap="Blues", shade=True, shade_lowest=False)

	plt.show()


"""
- Show point estimates and confidence intervals as rectangular bars.
- 双变量的分布图，注意事项：
	- y轴的变量类型必须是numeric;
	- 传入的estimator一般是来自numpy package, 比如np.mean, np.max, np.median
"""
def cate_biv_dis(data_frame, x_val, y_val, hue_name=None, estimator=None):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax = plt.subplots(figsize=a4_dims)

		# 判断绘图类型
		if hue_name is None:
			# 绘图
			if estimator is None:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, palette="Set3")
			else:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, estimator=estimator, palette="Set3")
		else:
			# 绘图
			if estimator is None:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, hue=hue_name, palette="Set3")
			else:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, hue=hue_name, estimator=estimator, palette="Set3")

		# 设置标题
		# fig.set_title('Bar Plot Of %s and %s'%(x_val, y_val), weight='bold')

	plt.show()


"""
- Figure-level interface for drawing categorical plots onto a FacetGrid.
- 和hue_val的区别在于：
	- 当hue_val取值较多时，比如超过5个，那么用带有hue_val的func画出来的图，就比较难以进行同hue取值之间的对比；
- 关键参数"kind":
	- # strip: 散点图, 相同取值的点会重合
    - # swarm: 分散点图，相同取值的点不会重合
    - # bar:条形图，条形是均值，线条的估计：线条的最上面的位置代表最大值的位置
    - # box: 箱线图
    - # count: 频次, 用这个kind，不需要传 y 参数
    - # violin: 小提琴图
	- Tricks:
		- 无y时，使用count;
		- 画散点图看聚合情况，用strip;
		- 两个变量用 box;
		- 可以加入hue_val; 
"""
def cate_facet_dis(data_frame, x_val, facet_val, hue_val, kind, y_val=None):
	with sns.axes_style("darkgrid"):
		# 绘图
		if y_val is None:
			fig = sns.catplot(x=x_val, col=facet_val, hue=hue_val, col_wrap=4, data=data_frame, kind=kind, size=2.5, aspect=.8, palette="Set3")
		else:
			fig = sns.catplot(x=x_val, y=y_val, col=facet_val, hue=hue_val, col_wrap=4, data=data_frame, kind=kind, size=2.5, aspect=.8, palette="Set3")
    
	plt.show()


"""
- Plot data and regression model fits across a FacetGrid. 
- 根据分面变量，加上分组变量调色，绘制单变量对因变量的线性拟合图
"""
def linear_facet(data_frame, x_val, y_val, facet_val):
	with sns.axes_style("darkgrid"): 
		# 多张画布，且每张画布的配色不一样
		fig = sns.lmplot(x=x_val, y=y_val, hue=facet_val, col=facet_val, data=data_frame, aspect=.4, x_jitter=.1) 

	plt.show()


"""
- Draw a plot of two variables with bivariate and univariate graphs.
- 双变量关系图
- Tricks:
	- 需要借用"scipy.stats.pearsonr()"来计算pearson相关系数;
	- 若想计算x和y的其它统计值，可以传对应的func给参数stat_func; 
	- 若觉得散点图不清晰，可以换成六角图, 传参kind = "hex"
"""
def cont_biv_rel(data_frame, val_1, val_2, stat_func=stats.pearsonr, kind="reg"):
	with sns.axes_style("darkgrid"): 
		ax = sns.jointplot(val_1, val_2, data=data_frame, kind=kind, stat_func=stat_func)

	plt.show()

