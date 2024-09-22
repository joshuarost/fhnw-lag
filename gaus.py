import numpy as np
from core import mrref


B=np.array([
    [ 0, 0, 2, 0, 0, 4],
    [ 1 ,1, -4, 1, 0, 2],
    [ 3, 3, -12, 3, 1, 7]])

print('Gauss-Jordan Elimination\n', mrref(B))
print('Schnittpunkt: ' ,B[:,-1])

