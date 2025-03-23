import numpy as np
import time
from sortedcontainers.sortedlist import SortedList
import heapq
from numba import njit

@njit
def find_k_smallest(distances, k):
    indices = np.full(k, -1)
    values = np.full(k, np.inf)
    
    for i in range(len(distances)):
        d = distances[i]
        max_idx = -1
        max_val = -np.inf
        for j in range(k):
            if values[j] > max_val:
                max_val = values[j]
                max_idx = j
        if d < max_val:
            values[max_idx] = d
            indices[max_idx] = i
    return values, indices


N = 1000000
d = 10
k = 100


X = np.random.uniform(-1,1,size=(N,d))

start_time = time.time()

distances = np.sum(X**2, axis=1)
output = find_k_smallest(distances, k)


end_time = time.time()

print(end_time - start_time)