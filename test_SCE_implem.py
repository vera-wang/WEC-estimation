#-------------------------------------------------------------------------------
# Name:        Testfile for Shuffled Complex Evolution Algorithm implementation
# Purpose:      Call one of the example functions to run and find the optimum
#               Visualize the 2D (or 3D) objective function +trace of the BESTX
#
#               if matplotlib is no available, comment the plot part
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import os
import sys
import time
import numpy as np
from keras.models import Sequential,load_model
#Call the scripts with SCE and
from SCE_python import *
from SCE_functioncall import *
from train_model import *
################################################################################
# PARAMETERS TO TUNE THE ALGORITHM
# Definition:
#  iseed = the random seed number (for repetetive testing purpose;pos integers)
#  iniflg = flag for initial parameter array (=1, included it in initial
#           population; otherwise, not included)
#  ngs = number of complexes (sub-populations)
# peps = value of NORMALIZED GEOMETRIC RANGE needed for convergence
#  maxn = maximum number of function evaluations allowed during optimization
#  kstop = maximum number of evolution loops before convergency
#  pcento = the percentage change allowed in kstop loops before convergency
#-------------------------------------------------------------------------------
maxn=500000
kstop=200
pcento=0.001
peps=0.001
iseed= 1
iniflg=1
ngs=35

ann_model = train()
#ann_model.save('chao2.h5')
ann_model = load_model('chao.h5')
#ann_model = load_model('ann.h5')
################################################################################

#foo=input('Please enter an Example number (1-5) for example, 6 for own model:')
start = time.clock()

################################################################################
# PARAMETERS FOR OPTIMIZATION PROBLEM
# Definition:
#  x0 = the initial parameter array at the start; np.array
#     = the optimized parameter array at the end;
#  f0 = the objective function value corresponding to the initial parameters
#     = the objective function value corresponding to the optimized parameters
#  bl = the lower bound of the parameters; np.array
#  bu = the upper bound of the parameters; np.array
################################################################################
foo = 6
if foo==1:
    '''1
     This is the Goldstein-Price Function
     Bound X1=[-2,2], X2=[-2,2]; Global Optimum: 3.0,(0.0,-1.0)
    '''
    bl=np.array([-2,-2])
    bu=np.array([2,2])
    x0=np.array([2,2])
    bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,testcase=True,testnr=1)
elif foo==2:
    '''2
      This is the Rosenbrock Function
      Bound: X1=[-5,5], X2=[-2,8]; Global Optimum: 0,(1,1)
        bl=[-5 -5]; bu=[5 5]; x0=[1 1];
    '''
    bl=np.array([-5.,-2.])
    bu=np.array([5.,8.])
    x0=np.array([-2.,7.])
    bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,testcase=True,testnr=2)

elif foo==3:
    '''3
    %  This is the Six-hump Camelback Function.
    %  Bound: X1=[-5,5], X2=[-5,5]
    %  True Optima: -1.031628453489877, (-0.08983,0.7126), (0.08983,-0.7126)
    '''

    bl=np.array([-2.,-2.])
    bu=np.array([2.,2.])
    x0=np.array([-1.,1.])
    bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,testcase=True,testnr=3)

elif foo==4:
    '''4
    %  This is the Rastrigin Function
    %  Bound: X1=[-1,1], X2=[-1,1]
    %  Global Optimum: -2, (0,0)
    '''
    bl=np.array([-5.,-5.])
    bu=np.array([5.,5.])
    x0=np.array([-1.,1.])
    bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,testcase=True,testnr=4)

elif foo==5:
    '''5
      This is the Griewank Function (2-D or 10-D)
      Bound: X(i)=[-600,600], for i=1,2,...,10  !for visualization only 2!
      Global Optimum: 0, at origin
    '''
    bl=-600*np.ones(2)
    bu=600*np.ones(2)
    x0=np.zeros(2)
    bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,testcase=True,testnr=5)

elif foo==6:
    '''6
    Run your own model - give
    '''
    bl=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
#    bu=np.array([60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60])
    bu=np.array([30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30])
    bu=bu*0.5
    #x0=np.array([1.31,1.24,1.13,1.52,2.54,1.65,5.11,3.82,0.66,1.59,1.07,0.84,0.88,6.18,1.00,4.31,1.02])

    mu = 0.5
    sigma = 1.0
    length_Sample=17
    s = np.random.normal(mu, sigma, length_Sample )
    #x0=np.power(10,s)
    x0=s

#    fun = lambda x: -sum([Qv[i]*x[i] for i in range(17)])
#    model = load_model('20191028210127.h5')
#    args = [1.42, 0.164, 0.544, 0.249, 0.147, 0.281, 0.545, 0.629, 0.337, 0.719, 0.139, 0.389, 0.372, 0.444, 0.5, 0.141, 1.134]
#    x0 = np.array([1.31,1.24,1.13,1.52,2.54,1.65,5.11,\
#                   3.82,0.66,1.59,1.07,0.84,0.88,6.18,1.00,4.31,1.02])
#    x0 = np.array([1.01，1.34，2.69，0.75，1.23，1.12，1.07，1.47，0.85，0.40，1.40，2.96，0.60，4.20，2.73，2.88，0.73])
    e = 1e-10

    
    bestx,bestf,BESTX,BESTF,ICALL = sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,ann_model,testcase=False,testnr=6)



elapsed = (time.clock() - start)
print ('The calculation of the SCE algorithm took %f seconds' %elapsed)

################################################################################
##  PLOT PART
################################################################################

'''
plot the trace of the parametersvalue
'''
import matplotlib.pyplot as plt
fig=plt.figure()
ax1=plt.subplot(121)
ax1.plot(BESTX)
plt.title('Trace of the different parameters')
plt.xlabel('Evolution Loop')
plt.ylabel('Parvalue')

'''
Plot the parmaeterspace in 2D with trace of the BESTX
'''
#- - - - - - - - - - - - - - - - - - - - - - - - -
#   make these smaller to increase the resolution
dx, dy = 0.05, 0.05
if foo == 5:
    print ('Use big enough steps to visualise parameter space')
##example 5 needs bigger steps!!!
##dx, dy = 5., 5.
#- - - - - - - - - - - - - - - - - - - - - - - - -

ax2=plt.subplot(122)
x = np.arange(bl[0], bu[0], dx)
y = np.arange(bl[1], bu[1], dy)
X,Y = np.meshgrid(x, y)

parspace=np.zeros((x.size,y.size))
for i in range(x.size):
    x1=x[i]
    for j in range(y.size):
        x2=y[j]
        parspace[i,j]=EvalObjF(x0.size,np.array([x1,x2]),ann_model,testcase=True,testnr=foo)

ax2.pcolor(X, Y, parspace)
##plt.colorbar()
ax2.plot(BESTX[:,0],BESTX[:,1],'*')
plt.title('Trace of the BESTX parameter combinations')
plt.xlabel('PAR 1')
plt.ylabel('PAR 2')

'''
Plot the parmaeterspace in 3D - commented out
'''
##from mpl_toolkits.mplot3d import Axes3D
##from matplotlib import cm
##from matplotlib.ticker import LinearLocator, FormatStrFormatter
##fig=plt.figure()
##ax = fig.gca(projection='3d')
##surf = ax.plot_surface(X, Y, parspace, rstride=8, cstride=8, cmap=cm.jet,linewidth=0, antialiased=False)
####cset = ax.contourf(X, Y, parspace, zdir='z', offset=-100)
##fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()