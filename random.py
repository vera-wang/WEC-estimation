# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
sampleNo = 300
length_Sample=17
log_Concentration=np.zeros([sampleNo,length_Sample])
for j in range(sampleNo):
    mu = 0.301029996
    sigma = 0.404148443
    #np.random.seed(598137259-203*j)
    #np.random.seed(598137259+2+203*j)
    np.random.seed(598137259+5+203*j)
    s = np.random.normal(mu, sigma, length_Sample )
    log_Concentration[j] = s
Concentration=np.power(10,log_Concentration)
#plt.subplot(111)
#plt.hist(s, 30, normed=True)