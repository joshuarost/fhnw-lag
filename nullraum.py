import numpy as np

from core import mnull, pp

C=np.array([
    [ 7 , 3 , -1],
    [  0, 4, 8,]])

pp('Nullraum', mnull(C))