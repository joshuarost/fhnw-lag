# -*- coding: utf-8 -*-
"""
Created on HS2022
@author: dadams
"""

import numpy as np
# symbolisches Rechnen mit u,v,x,y,z
#from sympy import *
import sympy as sym
import scipy.linalg as sp


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


# %% Die kleinste positive Zahl, die dargestellt werden kann:
print(np.finfo(float).eps)
# d.h. jede Zahl von dieser Grössenordnung betrachten wir als 0



#%% Polarkoordinaten zu kartesischen Koordinaten  NNHCXF 
# Geben Sie die kartesischen Koordinaten der Vektoren an. 
# [vx vy]=r*[cos(phi) sin(phi)]
# Befehle: np.array, np.cos, np.sin, print
# a)

#%% b)
r=5.0;phi=216.9  # in Grad


#%% b)
r = 13; phi= 0.4


#%% Normierung eines Vektors 5VRS99 
#1) Berechnen Sie die Komponenten von u=AB
#2) Berechnen Sie die Komponenten von v=u*1/|u|
#3) Berechnen Sie |v|
# Befehle: np.array, np.linalg.norm
A=np.array([ 3 , 4]) ; B=np.array([ 6, 0])

  

#%% Richtung und Länge  SWI49N
#Geben sie die kartesischen Koordinten der folgenden Vektoren an:
# Befehle: np.linalg.norm
# a) Länge 10
af=[8,-0.5] 

#%% b) Länge 5 
bs=[-33,56];



#%% Skalarprodukt, Orthogonalität  891584 
# Bestimme die Vektoren in der Liste, die zu v orthogonal sind.
# Befehle: np.dot oder np.matmul
v=[ 1, 5, 2 ];
a=[263, -35 ,-44]; b=[-121  ,15 , -48 ]; c=[71 ,5 ,-48 ];



#%% Schatten, spitzer/stumpfer Zwischenwinkel   QHZIHW, JBARLL 
# 1) Berechnen Sie die Länge des Schattens von b auf a 
# 2) Zwischenwinkel  0<phi<90° (spitz) oder 90<phi<180° (stumpf)?
# 3) Geben Sie den Schatten von b auf a  als Vektor an.
# Befehle: np.matmul, np.linalg.norm
#a)  
a=np.array([3,-4 ])/5;b=[12,-1.0]

# b) analog zu lösen
a=[4, 3];b=[ 12 ,-1]

# c)
a=[3 ,-4];b=[4 , 3]

#d)
a=[7, -24];b=[6.84, 5.12]




#%% Winkel zwischen Vektoren 520784 
# Berechne den Winkel zwischen den Vektoren a und b.
# Befehle: np.arccos, np.matmul, np.linalg.norm, np.pi
# a)
a=[1, 1, -1 ];b=[ 1, -1, 1 ]

# b) Geben Sie den Winkel in Grad an

# c)  Definieren Sie eine Funktion winkel(a,b), die den Winkel zwischen zwei Vektoren berechnet

#d) Berechnen Sie nun den Winkel zwischen a und b in Grad
a=[ 1, 1];b=[ 1, -1  ]


#%% Vektorprodukt in Orthonormalbasis   BT8J1D  
# Berechnen Sie die Vektorprodukte
# Befehle: np.cross
#%%a)
a=[-2 ,0 ,0] ; b=[0, 9 ,8] 

#%% b)    
a=[1,5,0 ]; b=[ 0,7,0 ]


#%% Fläche Dreieck   62FVCH 
#Berechne die Fläche des Dreiecks mit den Ecken A, B und C 
# Befehle: np.cross, np.linalg.norm(
#a)
A=np.array([  0 ,4 ,2 ]); B=[0 , 8 , 5 ]; C=[ 0 , 8, -1 ];


#%% Abstand Punkt-Gerade   CJ1IXZ  
# Abstand zwischen Gerade X=A+la*v und Punkt B
# Befehle: cross, norm, abs
#a)
A=np.array([3,0,0]);v=[2,0,0];B=[5,10, 0]


#%% Kollinear   RIMDII 
#Bestimme ob die Vektoren kollinear sind, indem du die erste Komponente eliminierst.
# Befehle: mrref
#a)
u=[-3,2];v=[ 2, -3];


#%%b)
u=[-3,2];v=[6,-4];


#%% Gauss-Verfahren: Gleichungen lösen  K9C5RL 
#Bestimmen Sie  die Dreiecksform mit dem Gaussverfahren.
# Lösen Sie dann das Gleichungssystem durch Einsetzen von unten nach oben.
# Befehle: np.matmul
#   x  - 4 y - 2 z =  -25  
#      - 3 y + 6 z  = -18
# 7 x - 13 y - 4 z =  -85

#   x     y     z  =   d 
mat=np.array([[  1 ,   - 4 , -2  , -25 ], 
             [ 0 ,   - 3 ,  6  , -18    ],
             [ 7  ,  - 13 , -4  , -85]])



#%% Komplanare Vektoren ACTUPR 
#Entscheide, ob die Vektoren komplanar sind. 
#Falls ja: Welche Linearkombination ergibt den Nullvektor $\vect{0}$?
# Befehle: np.array([a,b,c]).T , np.eye, mrref, np.concatenate
#a)
a=[3 ,2 ,0] ; b=[0, 4 ,3]; c=[3, 10, 6]
# print(mat)
# #   3     2     0     1     0     0
# #   0     4     3     0     1     0
# #   3    10     6     0     0     1
# mats=mrref(mat)
# [[ 1. 0. -0.5  0.41666667 0. -0.08333333]
#  [ 0. 1. 0.75 -0.125     0. 0.125 ]
#  [ 0. 0. 0.    0.5       1. -0.5 ]]
# # Die Vektoren sind linear abhängig
# # Die Linearkombination ist 0= 0.5*a + 1*b-0.5c




#%% b)
u=[3 ,0 ,-1] ; v=[ 0 , 4, 3] ; w=[15, -4 ,-7]

#%% c)
a=[3, 10, 6] ; b=[ 6 ,0 ,-3] ; c=[6 ,7 ,3]


#%%d)
u=[3 ,8, 5] ; v=[ 6, -4, -5] ; w=[6 ,12 ,7]


#%% Lösung eines LGS   L3YGQD  
# Überprüfen Sie, ob die angegebenen Lösungen die linearen Gleichungssysteme erfüllen. 
# Befehle: np.matmul
#a)        5 x + 2 y   = 65
u=[ 11 ,5];v=[ 7 ,14] ; w=[15 ,-5] 



#%% b) 5 x + 2 y  =5 
#   3 x + 2 y  =7 

u=[ 0 ,0] ; v=[ -1, 5] ; w=[ 5 ,-1]

#%% c) 5 x + 2 y   +z   =4 
#    x  + 3 y   +2z  =-1 
u=[ 3 , -18 ,25] ; v=[ 1 ,0 ,-1 ] ; w =[5 ,9, -1]

#%% d) -2 x        +z =-7 
#      x    +y  +2z=34 
#      x        + z=17
u=[0,1,17] ; v=[1,0,-1] ; w =[8,8,9]

  
#%% Gleichungen mit Parametern ITFSBL
#Bestimme x, so dass folgende Gleichungen erfüllt sind.
# a) u-v-x=x+v+x
# b) u-v-x=4x-v+x
# Befehle: sym.symboles, sym.solve

 