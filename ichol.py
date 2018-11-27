import numpy as np
import math
def mic(indptr: np.array,indices: np.array,data: np.array,n: int):
    for k in range(n):
        rowk = np.zeros(n)
        for j in range(indptr[k],indptr[k+1]):
            if indices[j] == k:
                data[j] = math.sqrt(data[j])
                rowk[indices[j]] = data[j]

            if indices[j] > k:
                if rowk[k] != 0:
                    data[j] = data[j] / rowk[k]
                rowk[indices[j]] = data[j]
        for i in range(k+1,n):
            for j in range(indptr[i],indptr[i+1]):
                data[j] = data[j] - rowk[indices[j]] * rowk[i]


