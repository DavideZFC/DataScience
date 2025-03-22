import numpy as np
import time

N = 1000000
d = 10

start_time = time.time()

X = np.random.uniform(-1,1,size=(N,d))

print(X.shape)