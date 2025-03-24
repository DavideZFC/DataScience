import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
import numpy as np
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
df = df.dropna()

y = df["availability_365"] > df["availability_365"].quantile(0.5)
X_raw = df[['minimum_nights','price','number_of_reviews']]
X_raw["room_neigh"] = df["room_type"] + " | " + df["neighbourhood_group"]

X_encoded = pd.get_dummies(X_raw, drop_first=True)
print(X_encoded.head())

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)
print(y_train)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')

display.plot()
plt.show()