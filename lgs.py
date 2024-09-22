import numpy as np

from core import mnull, mrref, pp

A=np.array([ 1, -4, 3, 2 ])

B=np.array([ [ 1, -8, -6, 1],
             [ 0, 1, 2, 2]])

C=np.array([ [ 1, 11, 0, 7, 5, -3],
             [ 3, 11, 5, 7, 0, 7],
             [ 4, 0, 5, 0, -5, -10]])

print("Aufgabe A")
Aes=mrref(B)
pp("Aufpunkt", Aes[:, -1])

riv=mnull(Aes[:,0:-1])
pp("Richtungsvektor", riv)

print("Aufgabe B")
Ces=mrref(C)
pp("Aufpunkt", Ces[:, -1])

riv=mnull(Ces[:,0:-1])
pp("Richtungsvektor", riv)