import numpy as np

A=[54, 6, -3]
B=[8, 10, -4]
C=[8, 6, -6]

P = [-28, -10, 6]
NULLPUNKT=[0, 0, 0]

# u = B - A
u = [2, -6, 0]
# u = [B[i] - A[i] for i in range(3)]
print('u ', u)

# v = C - A
v = [9, 0, 6]
# v = [C[i] - A[i] for i in range(3)]
print('v ', v)

# n = u x v
n = np.cross(u,v)
# n = [20, 21, 0]
print('Normalenvektor n ', n)

# w = P - A
w = [P[i] - A[i] for i in range(3)]

# Schatten h
h = np.dot(n,w)/np.linalg.norm(n)

print('Abstand: ', np.abs(h))

# Koordinatenform
kooef = [n[0], n[1], n[2], n[0]*A[0]+n[1]*A[1]+n[2]*A[2]]
print('Koordinatenform: ', kooef[0], 'x +', kooef[1], 'y +', kooef[2], 'z -', kooef[3], '= 0')