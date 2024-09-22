# %% BSP 9.30 Matrix mal Vektor = Einsetzen in Koeffizientenmatrix   XGQMA7
# Handelt es sich bei den Lösungen unten um partikuläre Lösungen, homogene Lösungen oder gar keine Lösung?                     \\
# Befehle: np.matmul np.array.T

# a)             x  y  z  = const , erweiterte Koeffizientenmatrix
import numpy as np


Aae = np.array([[4, 4, -16, 4], 
                [1, 0, -4, 2], 
                [5, 2, -20, 8]])
 

u = np.array([2, -1, 0])

v = [0, 2, 1]

w = [4, 0, 1]

koefizientenMatrix = Aae[:, :-1]
print(koefizientenMatrix)
print(np.matmul(koefizientenMatrix, u))
print(np.matmul(koefizientenMatrix, v))
print(np.matmul(koefizientenMatrix, w))