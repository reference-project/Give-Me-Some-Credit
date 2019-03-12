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
import statsmodels.api as sm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#导入数据
data = pd.read_csv('WoeDatanew.csv')
from sklearn.model_selection import train_test_split

Y = data['SeriousDlqin2yrs']
X = data.ix[:, 1:]
Xnew=X.drop(['DebtRatio','MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(Xnew, Y, test_size=0.3, random_state=0)


X1=sm.add_constant(X_train)
logit=sm.Logit(Y_train,X1)
result=logit.fit()
print(result.summary())
X3 = sm.add_constant(X_test)
resu = result.predict(X3)#进行预测
print(resu)

fpr, tpr, threshold = roc_curve(Y_test, resu)
rocauc = auc(fpr, tpr)#计算AUC
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)#生成ROC曲线
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('真正率')
plt.xlabel('假正率')
plt.show()


from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance,plot_tree,to_graphviz

lr = LogisticRegression()
result = lr.fit(X_train,Y_train)
probabilities1 = lr.predict_proba(X_train)
print('AUPRC = {}'.format(average_precision_score(Y_train,probabilities1[:, 1])))
print('AUC = {}'.format(roc_auc_score(Y_train,probabilities1[:, 1])))
probabilities2 = lr.predict_proba(X_test)
print('AUPRC = {}'.format(average_precision_score(Y_test,probabilities2[:, 1])))
print('AUC = {}'.format(roc_auc_score(Y_test,probabilities2[:, 1])))


# 取600分为基础分值，PDO为20（每高20分好坏比翻一倍），好坏比取20。
p = 20 / math.log(2)
q = 600 - 20 * math.log(20) / math.log(2)
baseScore = round(q + p * coe[0], 0)

#计算分数函数
def get_score(coe,woe,factor):
    scores=[]
    for w in woe:
        score=round(coe*w*factor,0)
        scores.append(score)
    return scores
coe = [-9.4554, 0.6410, 0.5115,, 1.0310, 1.7268, 1.0470]
# 各项部分分数
x1 = get_score(coe[1], woe1, p)
x2 = get_score(coe[2], woe2, p)
x3 = get_score(coe[3], woe3, p)
x7 = get_score(coe[4], woe7, p)
x9 = get_score(coe[5], woe9, p)

#根据变量计算分数
def compute_score(series,cut,score):
    list = []
    i = 0
    while i < len(series):
        value = series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        list.append(score[m])
        i += 1
    return list





