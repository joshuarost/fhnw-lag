import numpy as np

from core import pp


A = np.array([[3, 1, -10],
                [3, 0, 1],
                [1, 0, 5]])

pp('Determinante', np.linalg.det(A))
