import numpy as np

from core import mnull, mrref, pp

C=np.array([
    [ 0 ,-12 ,6],
    [ 2 ,6, -2],
    [ 4 ,-12, 8]])

C=np.array([
    [ 0 ,-2 ,4],
    [ 4 ,16, 21],
    [ 2 ,10, 6]])

pp("Eliminiert ", mrref(C))
pp('Schnittmenge ', mnull(C))
