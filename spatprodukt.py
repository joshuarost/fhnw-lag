import numpy as np
from core import pp

def spatprodukt(a,b,c):
    return np.dot(a, np.cross(b,c))

a=np.array( [ -22, 28, -22 ])
b=np.array( [ 5, -5, -5])
c=np.array( [-6, 4,-6] )

# A x B = (a2b3 - a3b2)i - (a1b3 - a3b1)j + (a1b2 - a2b1)k
print('Spatprodukt: ',spatprodukt(a,b,c))
print("Sind in einer Ebene? ", spatprodukt(a,b,c)==0)
