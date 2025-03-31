import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

data = pd.read_csv('datasets\AB_NYC_2019.csv')
print(data.info())

print(data.pivot_table("host_id", index='neighbourhood_group', columns='room_type', aggfunc='count'))

data = data[data['room_type'] != 'Shared room']
color_map = {'Entire home/apt': 'red', 'Private room': 'green'}
color = data['room_type'].map(color_map)


X_train, X_test, y_train, y_test = train_test_split(data[['latitude', 'longitude']], data['room_type'], train_size=0.9)

plt.scatter(X_test['latitude'], X_test['longitude'], color=y_test.map(color_map), s=0.2)
plt.show()

# SVM
n = X_train.shape[0]
print(X_train.std()*n)
m = 10
i = 1
n_samp = int((i+1)*n/m)

start = time.time()
svm = SVC(C=1000, gamma=1000)
svm.fit(X_train.iloc[:n_samp,:], y_train[:n_samp])
y_pred = svm.predict(X_test)
print('n samples', n_samp)

print("SVM time ", time.time()-start)

print("SVM accuracy", accuracy_score(y_test, y_pred))

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

'''
# XGBoost
start = time.time()
xgboost = xgb.XGBClassifier(tree_method="hist", max_depth=5, eval_metric='rmse')
label_map = {'Entire home/apt': 0, 'Private room': 1}
reverse_label_map = {0: 'Entire home/apt', 1: 'Private room'}
xgboost.fit(X_train, y_train.map(label_map))
y_pred = xgboost.predict(X_test)
y_pred = pd.Series(y_pred).map(reverse_label_map)
print("XGB time ", time.time()-start)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("XGB accuracy", accuracy_score(y_test, y_pred))
'''


