import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def sigmoid(x):
    return 1/(1+np.exp(-x))

N = 1000
k = 10
x = np.linspace(-np.pi, np.pi, num=N)

fourier_features = np.zeros((N,k))

for i in range(k):
    fourier_features[:, i] = np.cos(i*x)

columns = ["freq {}".format(i) for  i in range(k)]

data = pd.DataFrame(fourier_features, index=x, columns=columns)

# y = x**2
steepness = 6
y = steepness*np.cos(8*x)
y_norm = sigmoid(y)
y_rand = np.random.binomial(1, y_norm)

X_train, X_test, y_train, y_test = train_test_split(data, y_rand, train_size=0.9)

pca = PCA()
transform_df = pca.fit_transform(X_train)
plt.plot(pca.explained_variance_ratio_)
plt.show()

model = LogisticRegression(penalty='elasticnet', C=0.01, solver='saga', l1_ratio=0.01)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_pred_proba = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred_proba)
print(auc(fpr, tpr))

roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name='example estimator')
display.plot()

plt.show()
