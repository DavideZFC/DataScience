import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]

df["price_category"] = pd.qcut(df["price"], q=3, labels=["cheap", "medium", "expensive"])
print(df["price_category"])

features_to_use = ["neighbourhood_group", "room_type", "minimum_nights", "availability_365"]
X_raw = df[features_to_use]
X_encoded = pd.get_dummies(X_raw, drop_first=True)  # one-hot encode categoriche
y = df["price_category"]

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, train_size=0.9)
model = LogisticRegression(multi_class="multinomial")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('accuracy is '+str(acc))

repo = classification_report(y_test, y_pred)
print(repo)

confusion = confusion_matrix(y_test, y_pred)
print(confusion)

coeffs = pd.Series(model.coef_[0], index=X_encoded.columns)
print(coeffs.sort_values(ascending=False))
