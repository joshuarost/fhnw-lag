import numpy as np

def HesseNormal(x,y,coeff=[ 4 , 3,-6]):
    # Bestimmt Abstand zu Geraden 
    # in: 
    #     x,y: Koordinaten des Punktes
    #     coef: Koeffizienten der Gerade; default 4x+3y-6=0
    return np.dot( [x,y,1],coeff)/np.linalg.norm( coeff[0:len(coeff)-1])

# Abstand von Punkt zu Gerade
coeff1=[2, -1, -24]
print("Abstand: ", np.abs(HesseNormal(5,2 , coeff1)))