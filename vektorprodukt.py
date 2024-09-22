import numpy as np


vv=np.array( [1 ,3 ,5 ])
ww=np.array( [-1 ,7 ,1] )

# A x B = (a2b3 - a3b2)i - (a1b3 - a3b1)j + (a1b2 - a2b1)k
print('Vektorprodukt',np.cross(vv,ww))
