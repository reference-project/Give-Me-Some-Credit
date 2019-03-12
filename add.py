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
woe_data = pd.read_csv('D:/statistics/machine learning in action/GiveMeSomeCredit/WoeData.csv')

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

Y = data['SeriousDlqin2yrs']
X0 = data.ix[:, 10]
cutx0 = [float('-inf'), 0, 1, 2, 3, 5, float('inf')]

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
d40,iv0,woe0 = mono_bin2(Y,X0,cutx0)
print(iv0)


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



woe_data = pd.read_csv('D:/statistics/machine learning in action/GiveMeSomeCredit/WoeData.csv')
woe_data['NumberOfDependents'] = Series(replace_woe(woe_data['NumberOfDependents'], cutx0, woe0))
woe_data.to_csv('WoeDatanew.csv', index=False)