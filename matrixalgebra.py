import numpy as np

from core import pp

# MATRIXPRODUKT mit Falk-Schema
# A*B != B*A
# np.matmule(A, B) OR print(np.dot(A, B))
def matrixprodukt(A, B):
        try:
            return np.matmul(A, B)
        except ValueError:
              return "Dimensions Error! Matrixprodukt nicht möglich."

# Symmetrie check
def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

# AUFGABEN
A = np.array([[0, 1], 
    [2, 3], 
    [4, 2]])

B = np.array([[2, -1, -2], 
              [2, 3, -1]])

C = np.array([1, 0])

D = np.array([[-8, 0, -4],
             [4, 6, 3],
             [0, -3, -1]])

E = np.array([[0, -4, -3],
             [7, 9, 2],
             [-1, 0, -5]])

# TRANSPONIEREN
pp("A", A)
pp("A transponiert", A.T)

# SUMME
pp("+", np.add(D, E))

# MULTIPLIKATION
pp("*", np.multiply(E, D))
pp("2 *", np.multiply(2, A))

# MATRIXPRODUKT
pp("B", matrixprodukt(A, C))
pp("C", matrixprodukt(A, A.T))
pp("D", matrixprodukt(A, B))
pp("C.T", C.T)
pp("E", matrixprodukt(B, C.T))
