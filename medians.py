import numpy as np
import time
from numba import njit

@njit
def get_median(samples):
    partition_function = np.zeros(k)
    for i in range(N):
        partition_function[samples[i]:] += 1

    for i in range(k):
        if partition_function[i] == N /2:
            return i+0.5
        if partition_function[i] > N /2:
            return i

@njit
def get_fast_median(samples):
    density_function = np.zeros(k)
    for i in range(N):
        density_function[samples[i]] += 1
    
    partition_function = np.cumsum(density_function)
    for i in range(k):
        if partition_function[i] == N /2:
            return i+0.5
        if partition_function[i] > N /2:
            return i


N = 5000000
k = 1000

samples = np.random.randint(low=0,high=k,size=N)

start = time.time()
print(get_median(samples))
end = time.time()

print(end - start)

start = time.time()
print(get_fast_median(samples))
end = time.time()

print(end - start)