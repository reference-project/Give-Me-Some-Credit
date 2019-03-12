#!/usr/bin/env python 
# -*- coding:utf-8 -*-
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

#进行机器学习

##分割产生训练集和测试集
from sklearn.model_selection import train_test_split

Y = data['SeriousDlqin2yrs']
X = data.ix[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# print(Y_train)
train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)
"""train.to_csv('TrainData.csv',index=False)
test.to_csv('TestData.csv',index=False)"""

##模型 比较逻辑回归模型、决策树模型和随机森林模型的AUC值，择优选用。

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance,plot_tree,to_graphviz
from sklearn import tree
#使用决策树模型
"""weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
print(weights)
model = tree.DecisionTreeClassifier(max_depth=5)
model.fit(X_train, Y_train)
with open("credit.dot", 'w') as f:
    f = tree.export_graphviz(model, out_file=f)

import pydotplus
dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("credit.pdf")

probabilities1 = model.predict_proba(X_train)
print('AUPRC = {}'.format(average_precision_score(Y_train,probabilities1[:, 1])))
probabilities2 = model.predict_proba(X_test)
print('AUPRC = {}'.format(average_precision_score(Y_test,probabilities2[:, 1])))

def cv_score(d):
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, Y_train)
    probabilities1 = clf.predict_proba(X_train)
    probabilities2 = clf.predict_proba(X_test)
    return(average_precision_score(Y_test,probabilities2[:, 1]), average_precision_score(Y_train,probabilities1[:, 1]))

depths = np.arange(6,10)
scores = [cv_score(d) for d in depths]
tr_scores = [s[1] for s in scores]
te_scores = [s[0] for s in scores]

plt.figure(figsize=(6,4), dpi=120)
plt.grid()
plt.xlabel('max depth of decison tree')
plt.ylabel('AUPRC')
plt.plot(depths, te_scores, label='test_scores')
plt.plot(depths, tr_scores, label='train_scores')
plt.legend()
plt.show()"""

#使用随机森林
"""rf = RandomForestClassifier(random_state=0,n_estimators=200,max_depth=7,n_jobs=-1)
rf.fit(X_train,Y_train)
probabilities1 = rf.predict_proba(X_train)
print('AUPRC = {}'.format(average_precision_score(Y_train,probabilities1[:, 1])))
probabilities2 = rf.predict_proba(X_test)
print('AUPRC = {}'.format(average_precision_score(Y_test,probabilities2[:, 1])))"""

#使用逻辑回归
lr = LogisticRegression()
result = lr.fit(X_train,Y_train)
probabilities1 = lr.predict_proba(X_train)
print('AUPRC = {}'.format(average_precision_score(Y_train,probabilities1[:, 1])))
print('AUC = {}'.format(roc_auc_score(Y_train,probabilities1[:, 1])))
probabilities2 = lr.predict_proba(X_test)
print('AUPRC = {}'.format(average_precision_score(Y_test,probabilities2[:, 1])))
print('AUC = {}'.format(roc_auc_score(Y_test,probabilities2[:, 1])))

#使用Xgboost
weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())
clf = XGBClassifier(max_depth = 4, scale_pos_weight = weights,n_jobs = -1)
probabilities = clf.fit(X_train, Y_train).predict_proba(X_test)
print(probabilities)
print('AUPRC = {}'.format(average_precision_score(Y_test,probabilities[:, 1])))

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(clf, height=1, color=colours,grid=False,show_values=False, importance_type='cover', ax=ax)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

ax.set_xlabel('importance score', size=16)
ax.set_ylabel('features', size=16)
ax.set_yticklabels(ax.get_yticklabels(), size=12)
ax.set_title('Ordering of features by importance to the model learnt', size=20)
plt.show()

trainSizes, trainScores, crossValScores = learning_curve(XGBClassifier(max_depth = 4, scale_pos_weight = weights, n_jobs = -1), X_train,\
                                         Y_train, scoring = 'average_precision')
trainScoresMean = np.mean(trainScores, axis=1)
trainScoresStd = np.std(trainScores, axis=1)
crossValScoresMean = np.mean(crossValScores, axis=1)
crossValScoresStd = np.std(crossValScores, axis=1)

colours = plt.cm.tab10(np.linspace(0, 1, 9))

fig = plt.figure(figsize = (14, 9))
plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,
    trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])
plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,
    crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])
plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', \
         color = colours[0])
plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val', \
         color = colours[1])

ax = plt.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15), \
               loc=2, borderaxespad=0, fontsize = 16)
plt.xlabel('training set size', size = 16)
plt.ylabel('AUPRC', size = 16)
plt.title('Learning curves indicate slightly underfit model', size = 20)
plt.show()