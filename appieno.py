import numpy as np
import matplotlib.pyplot as plt

n = 1000
d = 2

S = np.array([[1,-1],[2,1]])
X0 = np.random.normal(0,1,size=(n,d))

X = np.dot(X0, S)


C = np.dot(X.T, X)/n

U, D, V = np.linalg.svd(C)
Y = np.dot(X, U)

print(np.dot(U[:,0], U[:,1]))


plt.gca().set_aspect('equal')
plt.scatter(X[:,0], X[:,1], alpha=0.3)

d1 = D[0]
d2 = D[1]

plt.arrow(0, 0, d1*U[0, 0], d1*U[1, 0])
plt.arrow(0, 0, d2*U[0, 1], d2*U[1, 1])
plt.show()