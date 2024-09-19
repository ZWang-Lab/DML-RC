#################################################################
# Code written by Gang Xu (g.xu@yale.edu)
# For bug issues, please contact author using the email address
#################################################################



import numpy as np
import pandas as pd
from math import sqrt
import pickle
import time
import math
from math import sin
from math import cos
from math import pi
from minimax_tilting_sampler import TruncatedMVN # for truncated MVNs
import statsmodels.api as sm
from scipy import stats
import copy

#Generate exposure variable and covariate
def ge_right(n,age_min,age_max,mean_x,sigma_x,sigma_ep):
#n: interger, sample size
#age_min: interger, the minimum value of the covariate age 
#age_max: interger, the maximum value of the covariate age 
#mean_x: vector, mean value of personal measurements
#sigma_x: matrix, covariance matrix of  of personal measurements
#sigma_ep: matrix, covariance matrix of  of measurements error

    ## personal measures
    mean_x=np.array(mean_x)
    lb = np.zeros_like(mean_x)
    ub = np.ones_like(mean_x) * np.inf
    sigma_x_copy=copy.copy(sigma_x)
    tmvn = TruncatedMVN(mean_x, sigma_x_copy, lb, ub)
    samples = tmvn.sample(n)
    per_x=samples.T   

    ## error terms
    mean_ep=np.zeros_like(mean_x)

    epsilon_x=np.random.multivariate_normal(mean_ep, sigma_ep, (n,), 'raise') 

    x1=np.array(per_x)
    x2=np.array(epsilon_x)

    
    age=np.random.uniform(age_min,age_max,n)
    race=np.random.binomial(1, 0.16, n)
    x1[:,1]=x1[:,1]+0.1*(race)

    ## nearest monitor measures
    x_nm=np.zeros_like(x1)
    for i in range(n):
        trig=0
        while trig==0:
            x_nm[i,:]=x1[i,:]+x2[i,:]

            if (x_nm[i,:].min()<0):
                epsilon_x=np.random.multivariate_normal(mean_ep, sigma_ep, (1,), 'raise')
                x2[i,:]=epsilon_x
            else:
                trig=1

    data3=np.concatenate([x1,x_nm,age.reshape(-1,1),race.reshape(-1,1)],axis=1)
    return data3

#generate exposure variable, covariate, and the outcome
def ge_main(n,age_min,age_max,mean_x,b_0,b_x,b_w,sd_xi,sigma_x,sigma_ep,linearg=True):
#n: interger, sample size
#age_min: interger, the minimum value of the covariate age 
#age_max: interger, the maximum value of the covariate age 
#mean_x: vector, mean value of personal measurements
#b_0: real number, intercept in the main model
#b_x: real number, exposure coefficient of interest in the main model
#b_w: vector,  covariate coefficient in the main model
#sd_xi: positive number, standard deviation of error term in the main model
#sigma_x: matrix, covariance matrix of  of personal measurements
#sigma_ep: matrix, covariance matrix of  of measurements error
#linearg: Boolean value, indicate the main model is linear (True) or not (False)

    ##  error in main model
    ep_xixi=np.random.normal(0,sd_xi,n)
    ## personal, nearest monitor
    data2=ge_right(n,age_min,age_max,mean_x,sigma_x,sigma_ep)
    ######################### 'g()' function
    mean_x=np.array(mean_x)
    n_ele=mean_x.shape[0]
    w=data2[:,np.concatenate([range(n_ele),[2*n_ele],[2*n_ele+1]],axis=0)]
    w = np.delete(w, 1, axis=1)
    if linearg==True:
        g=b_0+w@np.array(b_w)
    else:
        g=b_0+b_w[0]*np.exp(500*(w[:,0]-0.02))/(1+np.exp(500*(w[:,0]-0.02)))+b_w[4]*np.exp(20*(w[:,4]-0.4))/(1+np.exp(20*(w[:,4]-0.4)))+b_w[7]*np.exp(30*(w[:,7]-0.3))/(1+np.exp(30*(w[:,7]-0.3)))+b_w[-1]*(w[:,-1])

    ### generate Y
    Y=b_x*data2[:,1]+g+ep_xixi
    ### partial R2 of X
    R2_x=((b_x*data2[:,1]).var())/(Y.var())
    ### R2
    R2_f=((b_x*data2[:,1]+g).var())/(Y.var())
    ### partial R2 of 'g()' function
    R2_g=((g).var())/(Y.var())

    final=[Y,data2,R2_x,R2_f,R2_g]
    return final
