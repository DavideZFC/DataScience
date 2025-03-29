import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


data = pd.read_csv('datasets\AB_NYC_2019.csv')
print(data.info())

print(data.pivot_table("host_id", index='neighbourhood_group', columns='room_type', aggfunc='count'))

data = data[data['room_type'] != 'Shared room']
color_map = {'Entire home/apt': 'red', 'Private room': 'green'}
color = data['room_type'].map(color_map)


X_train, X_test, y_train, y_test = train_test_split(data[['latitude', 'longitude']], data['room_type'], train_size=0.9)

plt.scatter(X_test['latitude'], X_test['longitude'], color=y_test.map(color_map), s=0.2)
plt.show()

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("LDA accuracy", accuracy_score(y_test, y_pred))

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("QDA accuracy", accuracy_score(y_test, y_pred))

# Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred = log.predict(X_test)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("Logistic accuracy", accuracy_score(y_test, y_pred))

# Random forest
rf = RandomForestClassifier(n_estimators=20, max_depth=5)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("RF accuracy", accuracy_score(y_test, y_pred))

# XGBoost
xgboost = xgb.XGBClassifier(tree_method="hist", max_depth=5, eval_metric='rmse')
label_map = {'Entire home/apt': 0, 'Private room': 1}
reverse_label_map = {0: 'Entire home/apt', 1: 'Private room'}
xgboost.fit(X_train, y_train.map(label_map))
y_pred = xgboost.predict(X_test)
y_pred = pd.Series(y_pred).map(reverse_label_map)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("XGB accuracy", accuracy_score(y_test, y_pred))

# KNN

knn = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)  # puoi provare anche 5, 10, 20
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

plt.scatter(X_test['latitude'], X_test['longitude'], color=pd.Series(y_pred).map(color_map), s=0.2)
plt.show()

print("KNN accuracy:", accuracy_score(y_test, y_pred))



