import numpy as np
import time
from sortedcontainers.sortedlist import SortedList
import heapq
from numba import njit

@njit
def find_k_smallest(distances, k):
    indices = np.full(k, -1)
    values = np.full(k, np.inf)
    limit_distance = np.inf
    worst_index = 0
    
    for i in range(len(distances)):
        d = distances[i]
        if d < limit_distance:
            indices[worst_index] = i
            values[worst_index] = d

            worst_index = np.argmax(values)
            limit_distance = values[worst_index]
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