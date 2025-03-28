import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
df = df.dropna()

print(df.info())

df_regressor = df[['neighbourhood_group', 'latitude', 'longitude', 'room_type', 'number_of_reviews']]
X_encoded = pd.get_dummies(df_regressor)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y)

max_depths = 2+2*np.arange(5)
rmses = []
models = []
for m in max_depths:
    model = DecisionTreeRegressor(max_depth=m)
    model.fit(X_train, y_train)
    models.append(model)

    y_pred = model.predict(X_test)
    rmses.append(mean_squared_error(y_test, y_pred)**0.5)

rmses_forest = []
models_forest = []
for m in max_depths:
    model = RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=m)
    model.fit(X_train, y_train)
    models_forest.append(model)

    y_pred = model.predict(X_test)
    rmses_forest.append(mean_squared_error(y_test, y_pred)**0.5)


rmses_xgb = []
models_xgb = []
for m in max_depths:
    model = reg = xgb.XGBRegressor(tree_method="hist", max_depth=m, eval_metric='rmse')
    model.fit(X_train, y_train)
    models_xgb.append(model)

    y_pred = model.predict(X_test)
    rmses_xgb.append(mean_squared_error(y_test, y_pred)**0.5)



plt.plot(max_depths, rmses, label='trees')
plt.plot(max_depths, rmses_forest, label='forest')
plt.plot(max_depths, rmses_xgb, label='xgb')
plt.legend()
plt.show()

importances = models_forest[0].feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx], align='center')
plt.yticks(range(len(importances)), np.array(X_encoded.columns)[sorted_idx])
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()