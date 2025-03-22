import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
df = df.dropna()

print(df.info())

# predict price as function of other variables: is it useful to divide by neighbourhood?

y = df["price"]
X = df[['neighbourhood_group', 'latitude', 'longitude', 'room_type', 'minimum_nights', 'availability_365']]

X_encoded = pd.get_dummies(X, drop_first=True)

normalize = True
if normalize:
    X_encoded = (X_encoded - X_encoded.mean())/X_encoded.std()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, train_size=0.9, random_state=601)

alpha_list = 10.0**np.array([-10, 0, 1, 2, 3, 4])
for alpha in alpha_list:
    model = Ridge(alpha=alpha)
    model.fit(X_train, np.log1p(y_train))

    y_pred = model.predict(X_test)
    err = mean_absolute_error(y_test, np.expm1(y_pred))

    print("error for alpha = {} is {}".format(alpha, err))

plt.hist(np.log1p(df['price']))
plt.show()