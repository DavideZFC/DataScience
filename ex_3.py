import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

N = 1000
k = 5
x = np.linspace(-np.pi, np.pi, num=N)

fourier_features = np.zeros((N,k))

for i in range(k):
    fourier_features[:, i] = np.cos(i*x)

columns = ["freq {}".format(i) for  i in range(k)]
print(columns)

data = pd.DataFrame(fourier_features, index=x, columns=columns)

y = x**2


model = ElasticNet(alpha=0.10, l1_ratio=0.9)
X_train, X_test, y_train, y_tets = train_test_split(data, y, train_size=0.9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(mean_squared_error(y_tets, y_pred))
print(r2_score(y_tets, y_pred))

coef_df = pd.DataFrame(model.coef_, index=data.columns)
print(coef_df)
