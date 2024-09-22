# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:53:16 2024

@author: donat.adams
"""

# -*- coding: utf-8 -*-
"""
Created on HS2022
@author: dadams
"""

import numpy as np
# symbolisches Rechnen mit u,v,x,y,z
#from sympy import *
import sympy as sym
#import scipy.linalg as sp
import matplotlib.pyplot as plt


# eigene Funktionen
def eliminate(Aa_in, tolerance=np.finfo(float).eps*10., fix=False, verbos=0):
    # eliminates first row
    # assumes Aa is np.array, As.shape>(1,1)
    Aa = Aa_in
    Nn = len(Aa)
    # Mm = len(Aa[0,:])
    if (Nn < 2):
        return Aa
    else:
        if not fix:
            prof = np.argsort(np.abs(Aa_in[:, 0]))
            Aa = Aa[prof[::-1]]
        if np.abs(Aa[0, 0]) > tolerance:
            el = np.eye(Nn)
            el[0:Nn, 0] = -Aa[:, 0] / Aa[0, 0]
            el[0, 0] = 1.0 / Aa[0, 0]
            if (verbos > 50):
                print('Aa \n', Aa)
                print('el \n', el)
                print('pr \n', np.matmul(el, Aa))
            return np.matmul(el, Aa)
        else:
            return Aa


def FirstNonZero(lis):
    return next((i for i, x in enumerate(lis) if x), len(lis)-1)


def SortRows(Aa):
    inx = np.array(list(map(FirstNonZero, Aa)))
    #print('inx: ',inx,inx.argsort())
    return Aa[inx.argsort()]


def mrref(Aa_in, verbos=0):
    Aa = Aa_in*1.0
    Nn = len(Aa)
    kklist = np.arange(0, Nn - 1)
    #print('kklist', kklist)
    for kk in kklist:
        Aa[kk:, kk:] = eliminate(Aa[kk:, kk:], verbos=verbos-1)
    Aa = SortRows(Aa)
    Aa = np.flipud(Aa)
    # for kk in kklist:
    for kkh in kklist:
        kk = FirstNonZero(Aa[kkh, :])
        Aa[kkh::, kk::] = eliminate(Aa[kkh::, kk::], fix=True, verbos=verbos-1)
    return np.flipud(Aa)


def mnull(Aa,verbos=0):
    rg = np.linalg.matrix_rank(np.array(Aa))
    dm = Aa.shape[0]
    dn = Aa.shape[1]
    nd = dm-rg
    # Aao=np.concatenate((Aa.T,np.identity(dm)),1)
    Aao = np.concatenate((np.identity(dn)*0.0, np.identity(dn)*1.0),1)
    Aao[ 0:dn,0:dm] = Aa.T
    Aarr=mrref(Aao)    
    opt=Aarr[dm-nd:, dn:].T
    if (verbos>10):
        print(Aa.shape,opt.shape)
        print('Test: ',np.matmul(Aa,opt))
    return(opt) 


# Test mrref
# Aa=np.array([[2,2,1,4],[1,9,2,3]])
# print('eliminate: \n',eliminate(Aa,fix=True,verbos=0))
# Aa=np.array([[3,2,1,4],[1,9,2,3],[1,6,6,0]])
# Alos=mrref(Aa,verbos=0)
# print('mrref:',np.matmul(Aa[:,:3],Alos[:,-1])-Aa[:,-1])
#  init_printing(use_unicode=True)
# Aae = np.array(
#     [[3, 3, -8, 6, 0, 14], [1, 1, -4, 2, 1, 3], [5, 5, -20, 10, 3, 13]])
# Aae = np.array([[3, 3, -8, 6, 0, 14], [1, 1, -4, 2, 1, 3]])
# Aaes = mrref(Aae)
# print(Aaes)
# print(np.matmul(Aae[:, :-1], [10, 0, 2, 0, 1]))
# riv = mnull(Aae)
# print(riv)
# Aae = np.array([[2, 1, -2, 7], [1, 8, -4, 20], [-3, 6, 0, 6]])
# Aaes = mrref(Aae)
# print(Aaes)
# print(np.matmul(Aae[:,:-1],[10,0,2,0,1]))


#%% Grundoperationen \hcode{SXIVRK}
print(3+5,5-3,3*5,15/3,np.sqrt(9),np.pi,np.exp( 1))

#%% Betrag, Logarithmen und trigonometrische Funktionen \hcode{KVRGJR}
print('Betrag',np.abs(-3))
print('Vorzeichen', np.sign(-3))
print('Logarithmus',np.log(3))
print('Logarithmus Basis 10',np.log10(100))
print('Logarithmus Basis 3', np.log(27)/np.log(3))

print('Cosinus',np.cos(-3))
print('Sinus',np.sin(-3))
print('Tangens',np.tan(-3))

print('Arcus Cosinus ',np.arccos(-0.3))
print('Arcus Sinus ',np.arcsin(-0.3))
print('Arcus Tangens ',np.arctan(-3))

#%% Grundoperationen für Vektoren \hcode{MXG6YS}
vv=np.array( [1 ,3 ,5 ])
ww=np.array( [-1 ,7 ,1] )
print('Dimensionen',vv.shape )
print('Anz. Elemente',len(vv))
print('Norm',np.linalg.norm( vv),'oder',np.sqrt(np.dot(vv,vv)))
print('Skalarprodukt',np.dot(vv,ww))
print('Vektorprodukt',np.cross(vv,ww))

#%% Slicing \hcode{MT5TM5}
A=np.array([[1 ,3  ],[1 ,0  ],[ -1 ,7  ]])

print('Dimensionen',A.shape )
print('Anz. Zeilen',len(A ))

print('Auf einzelne Elemente zugreifen',A[0,1] )
A[0,1] =111
print('Einzelne Elemente neu definieren' )
print('2. Zeile',A[1,:] )
print('2. Spalte',A[:,1] )

print('letzte Spalte, 3. Zeile',A[2,-1] )
print('2. Spalte, letzte Zeile',A[-1,1] )


#%% Spezielle Matrizen \hcode{XFKJV4}
print('Einheitsmatrix \n',np.eye( 3 ))
print('Matrix mit 1 gefüllt \n',np.ones([3,2] ))
print('Matrix mit 0 gefüllt \n',np.zeros([3,2] ))
Dd=np.diag([3,1,2] )
print('Diagonalmatrix \n',Dd)
print('Die Diagonale einer Matrix extrahieren',np.diag(Dd))


#%% Operationen mit Arrays \hcode{QWJPJR}
aa=np.arange(1,4) 
bb=np.array( [3,5,7])
A=np.array([[1 ,3  ],[1 ,0  ],[ -1 ,7  ]])
B=np.array([[ -1 ,1 ,-1],[ -1 ,-1, 1],[ 1, -1 ,- 1]])
C=np.array([[ -1 ,1 ,-1],[ -1 ,-1, 1],[ -2,0,0]])
print('Die Array:',aa,bb)
print('Transposition:', A.T)
print('Matrixprodukt \n',np.dot(B ,A))
print('Elementweise Multiplikation',aa* bb,' und Division',bb/aa )
print('Elementweise Potenz',np.power(aa,2))

print('Determinante',np.linalg.det(B))
print('Inverse\n',np.linalg.inv(B))

print('Nullraum', mnull(C))
print('Gauss-Jordan Elimination\n', mrref(B))

print('Rang', np.linalg.matrix_rank(C))


#%% Symbolisches Rechnen \hcode{DC8BK4}
#from sympy import Symbol, solve

x = sym.Symbol('x', real=True)
print(sym.solve(-90 + 15*x + 15*x**2 , x, dict=True))

#%% Graphen von Funktioinen zeichnen 76B0UY
xlist = np.arange(-10., 10., 0.2)
def fun(x):
    return np.sin(x) 

def gun(x):
    return np.sin(-x)-2

ylist=list(map(fun,xlist))
yslist=list(map(gun,xlist))

# red dashes 'r--', yellow squares 'bs',  green triangles 'g^', blue  line 'b-'
# plt.plot( xlist, ylist, 'b-',xlist, yslist, 'g^')
# plt.xlabel('x [1]')
# plt.ylabel('y [1]')
# plt.show()

#%% Hessesche Normalenform GPNI6N
Pp=[ 10,4]  ; Qq= [ 11, 0] 
def HesseNormal(x,y,coeff=[ 4 , 3,-6]):
    # Bestimmt Abstand zu Geraden 
    # in: 
    #     x,y: Koordinaten des Punktes
    #     coef: Koeffizienten der Gerade; default 4x+3y-6=0
    return np.dot( [x,y,1],coeff)/np.linalg.norm( coeff[0:2])

print(HesseNormal(10,4) )
print(HesseNormal(11,0) )

# für die Gerade y= -3/4x +1/2 d.h. 3x+4y-2=0:
print(HesseNormal(10,4,[3,4,-2]) )
print(HesseNormal(11,0,[3,4,-2]) )

#%% Inhomogene LGS 8LTNE0
Ae=np.array([ [ 0, 3, -2, 3 ],[ 3, 0, -1, 6],[-2, 1 ,0 ,-3] ])
Aes=mrref(Ae)
print(Aes) # d.h. Aufpunkt ist [ 2,1,0]
print(Aes[:,0:-1])
riv=mnull( Aes[:,0:-1])
print(riv) # d.h. der Richtungsvektor ist  [ 1,2,3]

#%% Summen Y4AR4U
tac=np.arange(5,10+1)
print(tac)
suc=(3*tac)/np.power(tac+1,1/2)
suc=sum(suc)
print(suc)

#%% Linearität einer Funktion M27J2Z

u, v, w,x, y, z, lam = sym.symbols('u v w x y z lam')
sym.init_printing(use_unicode=True)
def lf(x,y,z):
    return sym.Array([ 5*x,-y])

# allgemein Ausdrücke vereinfachen
print(sym.simplify(sym.sin(x)**2 + sym.cos(x)**2))

# Homogenität; Resultat [0, 0] bedeutet, dass beide Ausdrücke gleich sind
print(sym.simplify(lf(lam*x,lam*y,lam*z)-lam*lf(x,y,z)))
# Additivität; Resultat [0, 0] bedeutet, dass beide Ausdrücke gleich sind
print( sym.simplify(lf( x , y,z )+lf(  u,  v,w)-lf( x+u, y+v,z+w)))

Ma=np.array([ lf(1,0,0),lf(0,1,0),lf(0,0,1) ])
print(Ma)