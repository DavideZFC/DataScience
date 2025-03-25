import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
df = df.dropna()

y = np.log1p(df['number_of_reviews'])
numerical = ['latitude', 'longitude', 'price', 'minimum_nights']
categoric = ['neighbourhood_group', 'room_type']

scaler = RobustScaler()
df_predict = pd.DataFrame(scaler.fit_transform(df[numerical]), columns=numerical, index=df.index)

df_predict = pd.concat([df_predict, df[categoric]], axis=1)

X_encoded = pd.get_dummies(df_predict, drop_first=True)
X_encoded['intercity'] = X_encoded['price']*X_encoded['neighbourhood_group_Brooklyn']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, train_size=0.2)

model = ElasticNet(alpha=0.001, l1_ratio=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

coef_df = pd.Series(model.coef_, index=X_encoded.columns)

print(coef_df)
print(mean_squared_error(y_test, y_pred)**0.5)
print(r2_score(y_test, y_pred))

vif_data = pd.DataFrame()
vif_data["feature"] = df_predict[numerical].columns
vif_data["VIF"] = [variance_inflation_factor(df_predict[numerical].values, i)
                   for i in range(df[numerical].shape[1])]

print(vif_data.sort_values("VIF", ascending=False))