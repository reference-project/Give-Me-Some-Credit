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
from scipy import stats
from pandas import Series


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
data['RevolvingUtilizationOfUnsecuredLines'].plot.box()
plt.show()
data = data[data['RevolvingUtilizationOfUnsecuredLines']<3000]
data['NumberOfTimes90DaysLate'].plot.box()
plt.show()
data = data.reset_index(drop = True)
print(data.info())
print(data.describe())

# 定义自动分箱函数
def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).values.argmax()
    return series.rank(pct=1).apply(f)

def mono_bin(Y, X, n = 20):
    r = 0
    good=Y.sum()
    bad=Y.count()-good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pct_rank_qcut(X, n)})
        d2 = d1.groupby('Bucket', as_index = True)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
        print(n)
    d3 = pd.DataFrame(d2.X.min(), columns = ['min'])
    d3['min']=d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe']=np.log((d3['rate']/(1-d3['rate']))/(good/bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by = 'min')).reset_index(drop=True)
    print("=" * 60)
    print(d4)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe

Y = data['SeriousDlqin2yrs']
X1 = data.ix[:, 1]
X2 = data.ix[:, 2]
X4 = data.ix[:, 4]
X5 = data.ix[:, 5]
X6 = data.ix[:, 6]
X7 = data.ix[:, 7]
X8 = data.ix[:, 8]
X9 = data.ix[:, 9]
X0 = data.ix[:, 10]
print(X0)
cutx6 = [float('-inf'), 1, 2, 3, 5, float('inf')]
cutx7 = [float('-inf'), 0, 1, 3, 5, float('inf')]
cutx8 = [float('-inf'), 0,1,2, 3, float('inf')]
cutx9 = [float('-inf'), 0, 1, 3, float('inf')]
cutx0 = [float('-inf'), 0, 1, 2, 3, 5, float('inf')]

d41, iv1, cutx1, woe1 = mono_bin(Y,X1)
d42, iv2, cutx2, woe2 = mono_bin(Y,X2)
d44, iv4, cutx4, woe4 = mono_bin(Y,X4)
d45, iv5, cutx5, woe5 = mono_bin(Y,X5)
X3 = data.ix[:, 3]
cutx3 = [float('-inf'), 0, 1, 3, 5, float('inf')]



def mono_bin2(Y,X,cut):
    good = Y.sum()
    bad = Y.count() - good
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": 0})
    for j in range(len(X)):
        for i in range(len(cut)):
            if X[j]>=cut[i] and X[j]<=cut[i+1]:
                d1['Bucket'][j] = i
                break
        print(i)
    d2 = d1.groupby('Bucket', as_index=True)
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = (d3.sort_index(by='min')).reset_index(drop=True)
    woe = list(d4['woe'].round(3))
    print(d4)
    return d4,iv,woe

d43,iv3,woe3 = mono_bin2(Y,X3,cutx3)
d46,iv6,woe6 = mono_bin2(Y,X6,cutx6)
d47,iv7,woe7 = mono_bin2(Y,X7,cutx7)
d48,iv8,woe8 = mono_bin2(Y,X8,cutx8)
d49,iv9,woe9 = mono_bin2(Y,X9,cutx9)
#d40,iv0,woe0 = mono_bin2(Y,X0,cutx0)
v10 = 0.034874597594118234



ivlist=[iv1,iv2,iv3,iv4,iv5,iv6,iv7,iv8,iv9,iv10]#各变量IV
index=['x1','x2','x3','x4','x5','x6','x7','x8','x9']#x轴的标签
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index))+1
ax1.bar(x, ivlist, width=0.4)#生成柱状图
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=14)
#在柱状图上添加数字标签
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.show()

#替换成woe函数
def replace_woe(series,cut,woe):
    list=[]
    i=0
    while i<len(series):
        value=series[i]
        j=len(cut)-2
        m=len(cut)-2
        while j>=0:
            if value>=cut[j]:
                j=-1
            else:
                j -=1
                m -= 1
        list.append(woe[m])
        i += 1
    return list

data['RevolvingUtilizationOfUnsecuredLines'] = Series(replace_woe(data['RevolvingUtilizationOfUnsecuredLines'], cutx1, woe1))
data['age'] = Series(replace_woe(data['age'], cutx2, woe2))
data['NumberOfTime30-59DaysPastDueNotWorse'] = Series(replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woe3))
data['DebtRatio'] = Series(replace_woe(data['DebtRatio'], cutx4, woe4))
data['MonthlyIncome'] = Series(replace_woe(data['MonthlyIncome'], cutx5, woe5))
data['NumberOfOpenCreditLinesAndLoans'] = Series(replace_woe(data['NumberOfOpenCreditLinesAndLoans'], cutx6, woe6))
data['NumberOfTimes90DaysLate'] = Series(replace_woe(data['NumberOfTimes90DaysLate'], cutx7, woe7))
data['NumberRealEstateLoansOrLines'] = Series(replace_woe(data['NumberRealEstateLoansOrLines'], cutx8, woe8))
data['NumberOfTime60-89DaysPastDueNotWorse'] = Series(replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woe9))
#data['NumberOfDependents'] = Series(replace_woe(data['NumberOfDependents'], cutx10, woex10))

data.to_csv('WoeData.csv', index=False)
