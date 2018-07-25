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
from scipy.stats import norm
import matplotlib.pyplot as plt

"""
- seaborn绘制图形基本总结；
- seaborn保存图片可以保存矢量图(svg)和非矢量图(jpg)，详细玩法见“方差分析”部分。
"""


"""
- Show the counts of observations in each categorical bin using bars.
- 单离散型变量的取值分布图，可以加上分类变量
"""
def cate_sig_dis(data_frame, col_name, hue_val=None):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax = plt.subplots(figsize=a4_dims)

		# 判断绘图类型
		if hue_val is None:
			# 绘图
			fig = sns.countplot(x=col_name, data=data_frame, palette="Set3")
		else:
			# 绘图
			fig = sns.countplot(x=col_name, hue=hue_val, data=data_frame, palette="Set3")

		# 设置标题
		# fig.set_title('Value Distribution Of %s'%(col_name), weight='bold')
	
	# 显示图片
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
		fig = sns.distplot(a=data_frame[data_frame[col_name].notnull()][col_name])

	# 显示图片
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

	# 显示图片
	plt.show()


"""
- Show point estimates and confidence intervals as rectangular bars.
- 双变量的分布图，注意事项：
	- y轴的变量类型必须是numeric;
	- 传入的estimator一般是来自numpy package, 比如np.mean, np.max, np.median
"""
def cate_biv_dis(data_frame, x_val, y_val, hue_val=None, estimator=None):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax = plt.subplots(figsize=a4_dims)

		# 判断绘图类型
		if hue_val is None:
			# 绘图
			if estimator is None:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, palette="Set3")
			else:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, estimator=estimator, palette="Set3")
		else:
			# 绘图
			if estimator is None:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, hue=hue_val, palette="Set3")
			else:
				fig = sns.barplot(x=x_val, y=y_val, data=data_frame, hue=hue_val, estimator=estimator, palette="Set3")

		# 设置标题
		# fig.set_title('Bar Plot Of %s and %s'%(x_val, y_val), weight='bold')

	# 显示图片
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

	# 显示图片
	plt.show()


"""
- Plot data and regression model fits across a FacetGrid. 
- 根据分面变量，加上分组变量调色，绘制单变量对因变量的线性拟合图
"""
def linear_facet(data_frame, x_val, y_val, facet_val):
	with sns.axes_style("darkgrid"): 
		# 多张画布，且每张画布的配色不一样
		fig = sns.lmplot(x=x_val, y=y_val, hue=facet_val, col=facet_val, data=data_frame, aspect=.4, x_jitter=.1) 

	# 显示图片
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

	# 显示图片
	plt.show()


"""
- 对角线上的图，就是单一连续变量的分布图，纵轴是坐标是迷惑。
- 只适合continuous变量间的对比。
- 可传入一个离散型变量作为hue_val。
- 特殊参数解析：【使用频率不高】
	- diag_kind：对角线的图形设定，默认是直方图。可以设为“kde”改为核密度图
	- kind：给非单变量图增加画图样式
"""
def cont_pair_analyse(data_frame, hue_val=None):
	with sns.axes_style("darkgrid"): 
		if hue_val is None:
			fig = sns.pairplot(data_frame, palette="Set3")
		else:
			fig = sns.pairplot(data_frame, hue=hue_val, palette="Set3")

	# 显示图片
	plt.show()


"""
- 多变量散点图
"""
def cont_biv_strip(data_frame, x_val, y_val):
	with sns.axes_style("darkgrid"): 
		f, ax = plt.subplots(figsize=a4_dims)
		fig = sns.regplot(x=x_val, y=y_val, data=data_frame, fit_reg=True)

	# 显示图片
	plt.show()


"""
- 箱线图
- 一般情况：
	- 横轴是离散型变量
	- 纵轴是连续型变量
- 当然横轴也可以是连续型变量，但或比较难看~
"""
def box_plot(data_frame, x_val, y_val):
	with sns.axes_style("darkgrid"):
		f, ax = plt.subplots(figsize=a4_dims)
		fig = sns.boxplot(x=x_val, y=y_val, data=data_frame, ax=ax, palette="Set3")

	# 显示图片
	plt.show()

"""
- 相关系数矩阵，用热力图呈现
- 热力图参数说明：
	- annot默认为False，当annot为True时，在heatmap中每个方格写入数据，annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等
	- fmt: String formatting code to use when adding annotations.
	- linewidths: Width of the lines that will divide each cell.
	- cbar: Whether to draw a colorbar.
	- square: If True, set the Axes aspect to “equal” so each cell will be square-shaped.
	- annot_kws: Keyword arguments for ax.text when annot is True.
	- linewidths: Width of the lines that will divide each cell. 相邻单元格之间的距离
"""
def corr_heat_map(data_frame, cor_thre=0.6, top_num=None, tar_val=None):
	# 计算相关系数矩阵
	corrmat = data_frame.corr()

	# 热力图是全部展示，还是只展示跟某个变量相关性比较强的变量
	if top_num is not None and tar_val is not None:
		cols_max = corrmat.nlargest(top_num, tar_val)[tar_val].index   # 取出与saleprice相关性最大的十项
		cols_min = corrmat.nsmallest(top_num, tar_val)[tar_val].index  # 取出与saleprice相关性最小的十项
		cols = list(cols_max) + list(cols_min)
		cols = list(set(cols))
		cm = np.corrcoef(data_frame[cols].values.T)  #相关系数
	else:
		cm = corrmat

	# 绘制热力图
	with sns.axes_style("darkgrid"):
		f, ax = plt.subplots(figsize=a4_dims) 
		hm_first = sns.heatmap(cm, linewidths=0.1, cbar=True, vmin=-1, vmax=1, center=0,
							  annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, 
							  yticklabels=cols, xticklabels=cols, cmap='RdBu')

	# 只突出一些特殊的数字
	for text in hm_first.texts:
		value_num = abs(float(text.get_text())) # 取绝对值
		if value_num >= cor_thre and value_num != 1.00:
			text.set_fontsize(8)
			text.set_weight('bold')
			text.set_color('orangered')
		else:
			text.set_alpha(0)

	# 设置标题
	if tar_val is not None:
		hm_first.set_title('Plot rectangular data as a color-encoded matrix of important variables:'%tar_val, weight='bold')
	else:
		hm_first.set_title('Plot rectangular data as a color-encoded matrix', weight='bold')

	# 显示图片
	plt.show()


"""
- 正态性检验
	- 条形图+正态曲线拟合
	- Q-Q图
	- 传入的变量格式类似：df_train['SalePrice']
"""
def cont_nor_dis(Series):
	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	# 画布1
	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax= plt.subplots(figsize=a4_dims)

		# 绘制正态曲线拟合
		fig = sns.distplot(Series, fit=norm, ax=ax) # fit 控制拟合的参数分布图形
		fig.set_title('Flexibly plot a univariate distribution of observations.', weight='bold') # 设置标题

	# 画布2
	with sns.axes_style("darkgrid"):
		# 设置画布大小
		f, ax= plt.subplots(figsize=a4_dims)

		# probplot :Calculate quantiles for a probability plot, and optionally show the plot. 计算概率图的分位数
		res = stats.probplot(Series, plot=plt) # seaborn画不了Q-Q图，这里使用states.probplot()来完成Q-Q图

	# 显示图片
	plt.show()

	# 如果遇到不符合正态分布的feature，可以使用log()转换
	# train_df['SalePrice'] = np.log(train_df['SalePrice'])


"""
- 批量画连续型变量的分布
"""
def multi_cont_dis(data_frame, cont_val_list, facet_num=4):
	#  检查传入的变量是否符合要求
	for col_name in cont_val_list:
		if data_frame.dtypes[col_name] == 'object':
			print 'Please Input Quantitative feature'
			sys.exit()

	with sns.axes_style("darkgrid"):
		f = pd.melt(data_frame, value_vars=cont_val_list)
		g = sns.FacetGrid(f, col="variable",  col_wrap=facet_num, sharex=False, sharey=False)
		g = g.map(sns.distplot, "value")

	# 显示图片
	plt.show()


"""
- 类别型变量对目标变量的单因素方差分析
- 方差分析的作用，stats.f_oneway()的文档解释：
	- If P < 0.05, we can claim with high confidence that the means of the results of all three experiments are significantly different.
	- p-value越小，表示差异越明显；
	- 为了更直观表示这种差异，做了单调性调转变换：np.log(1./a['pval'].values)
- 包含保存图片的玩法；
"""
def cate_anova(data_frame, tar_col):
	anv = pd.DataFrame()
	qualitative = [f for f in data_frame.columns if data_frame.dtypes[f] == 'object']
	anv['feature'] = qualitative
	pvals = []
	for c in qualitative:
		samples = []
		for cls in data_frame[c].unique():
			s = data_frame[data_frame[c] == cls][tar_col].values
			samples.append(s)
		pval = stats.f_oneway(*samples)[1]
		pvals.append(pval)
	anv['pval'] = pvals

	# 排序
	anv_sort = anv.sort_values('pval')

	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	# 画图
	a['disparity'] = np.log(1./a['pval'].values) # 调转单调性：p-value越小，柱状图的柱子越高， 表示差异越明显；
	with sns.axes_style("darkgrid"):
		f, ax= plt.subplots(figsize=a4_dims)
		sns_plot = sns.barplot(data=a, x='feature', y='disparity')
		ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)

		# 保存图片
		sns_plot.figure.savefig('cateVal_ANOVA_%s.svg'%(tar_col), format='svg', dpi=1200)
		sns_plot.figure.savefig('cateVal_ANOVA_%s.jpg'%(tar_col), dpi=100, bbox_inches='tight')

	# 显示图片
	plt.show()


"""
Demo:
- 连续型变量分桶后，观察对其它连续型变量的影响。
"""
def demo_cate_influ_cates():
	# load data
	df_train = pd.read_csv('../data/train.csv')
	df_test = pd.read_csv('../data/test.csv')

	# 对目标连续型feature进行log转换，使其正态性更好
	df_train['SalePrice'] = np.log(df_train['SalePrice'])

	# 合并数据集
	# 这里有个不提倡的使用：.loc[:,'MSSubClass':'SaleCondition']，这里等价于是要知道df_train的columns的排序，这么用非常不灵活
	all_df = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']), axis=0, ignore_index=True)

	# 过滤出连续型变量的列名
	quantitative = [f for f in all_df.columns if all_df.dtypes[f] != 'object']
	quantitative_features = quantitative
	train = all_df.loc[df_train.index]

	# 添加log转换后的目标feature
	train['SalePrice'] = df_train.SalePrice

	# 对目标feature进行分桶（这里的做法有些随意）
	standard = train[train['SalePrice'] < np.log(200000)]
	pricey = train[train['SalePrice'] >= np.log(200000)]

	# 新建一个dataFrame，存储结果
	diff = pd.DataFrame()
	diff['quantitative_feature'] = quantitative_features

	# 分析判断的逻辑相对简单哈
	difference_value = []
	for f in quantitative_features:
		difference_value.append((pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean())/(standard[f].fillna(0.).mean()))
	diff['difference'] = difference_value

	# 将画布设置为A4纸
	a4_dims = (11.7, 8.27)

	# 画图
	with sns.axes_style("darkgrid"):
		f, ax= plt.subplots(figsize=a4_dims)
		sns.barplot(data=diff, x='quantitative_feature', y='difference')
		x=plt.xticks(rotation=90)
	plt.show()
