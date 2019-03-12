#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor


train_data = pd.read_csv('D:/statistics/machine learning in action/GiveMeSomeCredit/cs-training.csv')

pd.set_option('display.max_columns', None)
print (train_data.head(5))
train_data.info()

print(train_data.describe())
print(train_data.shape)
del train_data['Unnamed: 0']

#数据预处理之填补缺失值MonsthlyIncome
def set_missing(df):
    newdata = df.iloc[ :,[5,0,1,2,3,4,6,7,8,9]]
    kown = newdata[newdata.MonthlyIncome.notnull()].as_matrix()
    unkown = newdata[newdata.MonthlyIncome.isnull()].as_matrix()
    X=kown[:,1:]
    Y=kown[:,0]
    rfr = RandomForestRegressor(random_state=0,n_estimators=200,max_depth=3,n_jobs=-1)
    rfr.fit(X,Y)
    predictions = rfr.predict(unkown[:, 1:]).round(0)
    print(predictions)
    df.loc[df.MonthlyIncome.isnull(), 'MonthlyIncome'] = predictions
    return df
data = set_missing(train_data)

#去掉缺失NumberOfDependents的行
data = data.dropna()
data = data.drop_duplicates()
print(data.shape)

#去掉不合理的异常值
data = data[data.age>0]
data = data[data['NumberOfTime30-59DaysPastDueNotWorse']<90]

print(data.info())
print(data.describe())

#探索数据分布及其与结果间的相关关系
# 使用cut函数，将连续变量转换成分类变量
def binning(col, cut_points, labels=None, isright=True):
    val_min = col.min()
    val_max = col.max()
    break_points = [val_min] + cut_points + [val_max]
    if not labels:
        labels = range(len(cut_points) + 1)
    else:
        labels = [str(i + 1) + ':' + labels[i] for i in range(len(cut_points) + 1)]
    colbin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True, right=isright)
    return colbin

# RevolvingUtilizationOfUnsecuredLines
df_tmp = data[['SeriousDlqin2yrs', 'RevolvingUtilizationOfUnsecuredLines']]
cut_points = [0.25, 0.5, 0.75, 1, 2]
labels = ['below0.25', '0.25-0.5', '0.5-0.75', '0.75-1.0', '1.0-2.0', 'above2']
df_tmp['Utilization_Bin'] = binning(df_tmp['RevolvingUtilizationOfUnsecuredLines'], cut_points, labels)
#print(df_tmp['Utilization_Bin'])
df_tmp.groupby(['Utilization_Bin'])['Utilization_Bin'].count().plot.bar()
plt.show()
df_tmp.groupby(['Utilization_Bin','SeriousDlqin2yrs'])['SeriousDlqin2yrs'].count()
df_tmp[['Utilization_Bin','SeriousDlqin2yrs']].groupby(['Utilization_Bin']).mean().plot.bar()
plt.show()

#age
data.groupby(['age'])['age'].count().plot.bar()
plt.show()
df_tmp = data[['SeriousDlqin2yrs', 'age']]
cut_points = [25, 35, 45, 55, 65]
labels = ['below25', '26-35', '36-45', '46-55', '56-65', 'above65']
df_tmp['age_Bin'] = binning(df_tmp['age'], cut_points, labels)
df_tmp.groupby(['age_Bin'])['age_Bin'].count().plot.bar()
plt.show()
df_tmp[['age_Bin','SeriousDlqin2yrs']].groupby(['age_Bin']).mean().plot.bar()
plt.show()

#Debtratio
df_tmp = data[['SeriousDlqin2yrs', 'DebtRatio']]
cut_points=[5,10,15,20,25,30]
labels=['below 5', '6-10', '11-15','16-20','21-25','26-30','above 30']
df_tmp['DebtRatio_Bin'] = binning(df_tmp['DebtRatio'], cut_points, labels)
df_tmp.groupby(['DebtRatio_Bin'])['DebtRatio_Bin'].count().plot.bar()
plt.show()
df_tmp[['DebtRatio_Bin','SeriousDlqin2yrs']].groupby(['DebtRatio_Bin']).mean().plot.bar()
plt.show()

#MonthlyIncome
df_tmp = data[['SeriousDlqin2yrs', 'MonthlyIncome']]
cut_points=[5000,10000,15000]
labels=['below 5000', '5000-10000','1000-15000','above 15000']
df_tmp['MonthlyIncome_Bin'] = binning(df_tmp['MonthlyIncome'], cut_points, labels)
df_tmp.groupby(['MonthlyIncome_Bin'])['MonthlyIncome_Bin'].count().plot.bar()
plt.show()
df_tmp[['MonthlyIncome_Bin','SeriousDlqin2yrs']].groupby(['MonthlyIncome_Bin']).mean().plot.bar()
plt.show()

#NumberRealEstateLoansOrLines
df_tmp = data[['SeriousDlqin2yrs', 'NumberRealEstateLoansOrLines']]
cut_points=[5,10,15,20]
labels=['below 5', '6-10', '11-15','16-20','above 20']
df_tmp['NumberRealEstateLoansOrLines_Bin'] = binning(df_tmp['NumberRealEstateLoansOrLines'], cut_points, labels)
df_tmp.groupby(['NumberRealEstateLoansOrLines_Bin'])['NumberRealEstateLoansOrLines_Bin'].count().plot.bar()
plt.show()
df_tmp[['NumberRealEstateLoansOrLines_Bin','SeriousDlqin2yrs']].groupby(['NumberRealEstateLoansOrLines_Bin']).mean().plot.bar()
plt.show()

#NumberOfDependents
data.groupby(['NumberOfDependents'])['NumberOfDependents'].count().plot.bar()
plt.show()
df_tmp = data[['SeriousDlqin2yrs', 'NumberOfDependents']]
cut_points = [1,2,3,4,5]
labels = ["0","1","2","3","4","5 and more"]
df_tmp['NumberOfDependents_Bin'] = binning(df_tmp['NumberOfDependents'], cut_points, labels)
df_tmp.groupby(['NumberOfDependents_Bin'])['NumberOfDependents_Bin'].count().plot.bar()
plt.show()
df_tmp[['NumberOfDependents_Bin','SeriousDlqin2yrs']].groupby(['NumberOfDependents_Bin']).mean().plot.bar()
plt.show()

#NumberOfTime30-59DaysPastDueNotWorse
data.groupby(['NumberOfTime30-59DaysPastDueNotWorse'])['NumberOfTime30-59DaysPastDueNotWorse'].count().plot.bar()
plt.show()
data[['NumberOfTime30-59DaysPastDueNotWorse','SeriousDlqin2yrs']].groupby(['NumberOfTime30-59DaysPastDueNotWorse']).mean().plot.bar()
plt.show()

#NumberOfTime60-89DaysPastDueNotWorse
data.groupby(['NumberOfTime60-89DaysPastDueNotWorse'])['NumberOfTime60-89DaysPastDueNotWorse'].count().plot.bar()
plt.show()
data[['NumberOfTime60-89DaysPastDueNotWorse','SeriousDlqin2yrs']].groupby(['NumberOfTime60-89DaysPastDueNotWorse']).mean().plot.bar()
plt.show()

#NumberOfTimes90DaysLate
data.groupby(['NumberOfTimes90DaysLate'])['NumberOfTimes90DaysLate'].count().plot.bar()
plt.show()
data[['NumberOfTimes90DaysLate','SeriousDlqin2yrs']].groupby(['NumberOfTimes90DaysLate']).mean().plot.bar()
plt.show()

#探索特征间的相关关系
sns.heatmap(data.corr())
plt.show()

#进行机器学习

##分割产生训练集和测试集
from sklearn.model_selection import train_test_split

Y = data['SeriousDlqin2yrs']
X = data.ix[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# print(Y_train)
train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)
train.to_csv('TrainData.csv',index=False)
test.to_csv('TestData.csv',index=False)

















