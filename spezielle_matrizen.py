import numpy as np

from core import pp
from matrixalgebra import matrixprodukt

A = np.array([[2, -1, -2], 
              [2, 3, -1]])

Y_SPIEGEL = np.array([[ -1, 0], 
                    [0, 1]])

X_SPIEGEL = np.array([[1, 0], 
                    [0, -1]])

PUNKTSPIEGELUNG_0 = np.array([[-1, 0],
                            [0, -1]])

# WEITERE SKRIPT S.213


pp("Speigelung", matrixprodukt(Y_SPIEGEL, A))