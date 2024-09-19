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
import statsmodels.api as sm
from doubleml import DoubleMLData
from doubleml import DoubleMLPLR
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from scipy import stats
from doubleml.datasets import make_plr_CCDDHNR2018
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import multiprocessing as mp
from sklearn.preprocessing import StandardScaler 

import copy

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import (Exchangeable, Independence,
                                           Nested, Autoregressive)

def var_est_dml(dml_model):
    (x, y, d) = (dml_model._dml_data.x,dml_model._dml_data.y,dml_model._dml_data.d)
    theta_hat=dml_model.coef[0]
    psi_a=dml_model.psi_elements['psi_a']
    psi=dml_model.psi

    model_g_predict=((dml_model.predictions)["ml_g"]).reshape(-1,1)
    model_m_predict=((dml_model.predictions)["ml_m"]).reshape(-1,1)

    if dml_model.apply_cross_fitting:
        _var_scaling_factor = dml_model._dml_data.n_obs
        y_test=np.array(y)
        d_test=np.array(d)
        x_test=np.array(x)
    else:
        smpls = dml_model._smpls
        test_index =smpls[0][0][1]
        _var_scaling_factor = len(test_index)
        psi_a=psi_a[test_index]
        psi = psi[test_index]
        model_g_predict=model_g_predict[test_index]
        model_m_predict=model_m_predict[test_index]
        y_test=np.array(y)[test_index]
        d_test=np.array(d)[test_index]
        x_test=np.array(x)[test_index,:]
    w1_x=np.concatenate((d_test.reshape(-1,1),x_test),axis=1)
    w1_x_add_cons=np.array(sm.add_constant(w1_x))
    J = np.mean(psi_a)
    if dml_model.score=="IV-type":
        sigma2_hat = 1 / _var_scaling_factor * np.mean(np.power(psi, 2)) / np.power(J, 2)
    else:
        sigma2_hat = np.linalg.inv(np.dot(w1_x_add_cons.T,w1_x_add_cons))[1,1]*np.sum(np.power(psi.reshape(-1)/d_test,2))/(_var_scaling_factor-x_test.shape[1]-2)
    return sqrt(sigma2_hat)

def var_est_gee(dml_model,using_lasso,w1_x_pred_per_add_cons, xw_nm_m,covmatrix_all,gee_params):
    (x, y, d) = (dml_model._dml_data.x,dml_model._dml_data.y,dml_model._dml_data.d)
    theta_hat=dml_model.coef[0]
    psi_a=dml_model.psi_elements['psi_a']
    psi=dml_model.psi

    model_g_predict=((dml_model.predictions)["ml_g"]).reshape(-1,1)
    model_m_predict=((dml_model.predictions)["ml_m"]).reshape(-1,1)

    if dml_model.apply_cross_fitting:
        _var_scaling_factor = dml_model._dml_data.n_obs
        y_test=np.array(y)
        d_test=np.array(d)
        x_test=np.array(x)
        nm_xd_test=xw_nm_m
    else:
        smpls = dml_model._smpls
        test_index =smpls[0][0][1]
        _var_scaling_factor = len(test_index)
        psi_a=psi_a[test_index]
        psi = psi[test_index]
        model_g_predict=model_g_predict[test_index]
        model_m_predict=model_m_predict[test_index]
        y_test=np.array(y)[test_index]
        d_test=np.array(d)[test_index]
        x_test=np.array(x)[test_index,:]
        nm_xd_test=xw_nm_m[test_index,:]
    w1_x=np.concatenate((d_test.reshape(-1,1),x_test),axis=1)
    w1_x_add_cons=np.array(sm.add_constant(w1_x))
    J = np.mean(psi_a)
    if dml_model.score=="IV-type":
        sigma2_hat = 1 / _var_scaling_factor * np.mean(np.power(psi, 2)) / np.power(J, 2)
    else:
        sigma2_hat = np.linalg.inv(np.dot(w1_x_add_cons.T,w1_x_add_cons))[1,1]*np.sum(np.power(psi.reshape(-1)/d_test,2))/(_var_scaling_factor-x_test.shape[1]-2)

    devpsi=[]
    devpsi0=np.zeros(covmatrix_all.shape[0])
    ele_total=int(len(gee_params)/xw_nm_m.shape[1])
    ele_num=0
    for ele_num in range(ele_total):
        index=np.array(range(xw_nm_m.shape[1]))+(ele_num)*(xw_nm_m.shape[1])
        x1=xw_nm_m
        sum_all=np.zeros(x1.shape[1])
        if(using_lasso==True):    
            for k in range(x1.shape[1]):
                params=copy.deepcopy(gee_params[index])
                params[k]=params[k]+0.0001
                dml_model_copy=copy.deepcopy(dml_model)
                if(ele_num==0):
                    pred_d=np.dot(x1,params)
                    for num in range(len(dml_model_copy._dml_data.d)):
                        dml_model_copy._dml_data.d[num]=pred_d[num]
                else:
                    dml_model_copy._dml_data.x[:,ele_num-1]=np.dot(x1,params)
                dml_model_copy.fit(store_predictions=True,store_models=True)
                g_pred=((dml_model_copy.predictions)["ml_g"]).reshape(-1)
                m_pred=((dml_model_copy.predictions)["ml_m"]).reshape(-1)
                d_new=dml_model_copy._dml_data.d
                dml_model_new_psi = dml_model_copy.psi
                if dml_model.apply_cross_fitting:
                    d_new_test=np.array(d_new)
                else:
                    d_new_test=np.array(d_new)[test_index]
                    g_pred=g_pred[test_index]
                    m_pred=m_pred[test_index]
                    dml_model_new_psi = dml_model_copy.psi[test_index]

                if dml_model.score=="IV-type":
                    new_psi=(d_new_test-m_pred)*(y_test-(d_new_test)*theta_hat-g_pred)
                else:
                    new_psi=(d_new_test)*(y_test-(d_new_test)*theta_hat-g_pred)
                for kn in range(len(y_test)):
                    sum_all[k]=sum_all[k]+(new_psi)[kn]-psi[kn]
            sum_all=sum_all*10000
        devpsi=np.concatenate((devpsi,sum_all), axis=None)
    var_nmall=np.dot(devpsi,np.dot(covmatrix_all,devpsi))/nm_xd_test.shape[0]
    var_nmD=np.dot(devpsi0,np.dot(covmatrix_all,devpsi0))/nm_xd_test.shape[0]
    var_total2=sigma2_hat+ var_nmall / np.power(np.sum(psi_a), 2)


    return  sqrt(sigma2_hat+1 / _var_scaling_factor * var_nmD / np.power(J, 2)),sqrt(var_total2)

def data_me_gee(x_per_v,x_nm_v,w1_per_v,w1_nm_v,w2_v):
    x_per_v=np.array(x_per_v).reshape(-1)
    w1_per_v=np.array(w1_per_v)
    x_nm_v=np.array(x_nm_v).reshape(-1,1)
    w1_nm_v=np.array(w1_nm_v)
    xw_nm_v=np.concatenate((x_nm_v,w1_nm_v,w2_v),axis=1)
    I=np.identity(1+w1_per_v.shape[1])
    xw_nm_v =sm.add_constant(xw_nm_v)
    x_all=np.kron(I,xw_nm_v)

    y_all=x_per_v
    id_all=range(len(y_all))
    for i in range(w1_per_v.shape[1]):
        y_all=np.concatenate((y_all,w1_per_v[:,i]),axis=0)
        id=range(len(w1_per_v[:,i]))
        id_all=np.concatenate((id_all,id),axis=0)

    cov_struct = Independence()
    gee_model = sm.GEE(y_all, x_all, groups=id_all, cov_struct=cov_struct)
    gee_results = gee_model.fit()

    sigma=np.zeros((w1_per_v.shape[1]+1,w1_per_v.shape[1]+1))
    unique_id=np.unique(id_all)
    for j in unique_id:
        yj=y_all[id_all==j]
        xj=x_all[id_all==j,:]
        res_1=(yj-xj@gee_results.params).reshape(-1,1)
        sigma=sigma+res_1@res_1.T
    sigma=sigma/(len(unique_id)-1)
    sigma_inv=np.linalg.inv(sigma)

    I22=np.zeros((x_all.shape[1],x_all.shape[1]))
    C22=np.zeros((x_all.shape[1],x_all.shape[1]))
    for j in unique_id:
        yj=y_all[id_all==j]
        xj=x_all[id_all==j,:]
        I22=I22+xj.T@sigma_inv@xj
        res_vec=xj.T@sigma_inv@(yj-xj@gee_results.params).reshape(-1,1)
        C22=C22+res_vec@res_vec.T
    I22_inv=np.linalg.inv(I22)
    var22=I22_inv@C22@I22_inv
    data = list()
    data.append(gee_results.params)
    data.append(var22)
    return data

def gee_predict(x_nm_m,w1_nm_m,w2_m,gee_params):
    x_nm_m=np.array(x_nm_m).reshape(-1,1)
    w1_nm_m=np.array(w1_nm_m)
    xw_nm_m=np.concatenate((x_nm_m,w1_nm_m,w2_m),axis=1)
    xw_nm_m=sm.add_constant(xw_nm_m)
    x_pred_per=xw_nm_m@gee_params[0:xw_nm_m.shape[1]]
    w1_pred_per=np.zeros(w1_nm_m.shape)
    for n in range(w1_pred_per.shape[1]):
        index=np.array(range(xw_nm_m.shape[1]))+(n+1)*(xw_nm_m.shape[1])
        w1_pred_per[:,n]=xw_nm_m@gee_params[index]

    return (x_pred_per,w1_pred_per,xw_nm_m)





def non_orth_score(y, d, l_hat, m_hat, g_hat, smpls):
    u_hat = y - g_hat
    psi_a = -np.multiply(d, d)
    psi_b = np.multiply(d, u_hat)
    return psi_a, psi_b



def dml(data,ml_l,ml_m,ml_g,true_para, ME,using_lasso, w1_x_pred_per_add_cons, xw_nm_m,gee_var,gee_params):
    (x, y, d) = data

    obj_dml_data = DoubleMLData.from_arrays(x, y, d)
    obj_dml_plr_nonorth_nosplit = DoubleMLPLR(obj_dml_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds=1,
                                      apply_cross_fitting=False,
                                      score=non_orth_score)
    obj_dml_plr_nonorth_nosplit.fit(store_predictions=True,store_models=True)
    this_theta_nonorth_nosplit = obj_dml_plr_nonorth_nosplit.coef[0]
    se_nonorth_nosplit=obj_dml_plr_nonorth_nosplit.se[0]
    se_nonorth_nosplit_v2=var_est_dml(obj_dml_plr_nonorth_nosplit)
    


    obj_dml_plr = DoubleMLPLR(obj_dml_data,
                              ml_l, ml_m, ml_g,
                              n_folds=2,
                              score='IV-type')
    obj_dml_plr.fit(store_predictions=True,store_models=True)
    this_theta_dml_plr = obj_dml_plr.coef[0]
    se_orth=obj_dml_plr.se[0]
    se_orth_v2=var_est_dml(obj_dml_plr)

    if ME:
        se_nonorth_nosplit1,se_nonorth_nosplit2=var_est_gee(obj_dml_plr_nonorth_nosplit,using_lasso,w1_x_pred_per_add_cons, xw_nm_m,gee_var,gee_params)
        se_orth1,se_orth2=var_est_gee(obj_dml_plr,using_lasso,w1_x_pred_per_add_cons, xw_nm_m,gee_var,gee_params)
        results=[this_theta_nonorth_nosplit-true_para,this_theta_dml_plr-true_para,se_nonorth_nosplit1,se_orth1,se_nonorth_nosplit2,se_orth2]
    else:
        results=[this_theta_nonorth_nosplit-true_para,this_theta_dml_plr-true_para,se_nonorth_nosplit,se_orth,se_nonorth_nosplit_v2,se_orth_v2]
    return results


def dml_lasso(data,true_para, ME,using_lasso, w1_x_pred_per_add_cons, xw_nm_m,gee_var,gee_params):
    ml_l = linear_model.Lasso(max_iter=100000, warm_start=True)
    ml_m = linear_model.Lasso(max_iter=100000, warm_start=True)
    ml_g = linear_model.Lasso(max_iter=100000, warm_start=True)
    (x, y, d) = data
    obj_dml_data = DoubleMLData.from_arrays(x, y, d)
    obj_dml_plr_nonorth_nosplit = DoubleMLPLR(obj_dml_data,
                                      ml_l, ml_m, ml_g,
                                      n_folds=1,
                                      apply_cross_fitting=False,
                                      score=non_orth_score)
    par_grids = {'ml_l': {'alpha': np.arange(0.001, 0.1, 0.005)},
                 'ml_m': {'alpha': np.arange(0.001, 0.1, 0.005)},
                 'ml_g': {'alpha': np.arange(0.001, 0.1, 0.005)}}
    obj_dml_plr_nonorth_nosplit.tune(par_grids, search_mode='grid_search');
    print("ML_full_sample",obj_dml_plr_nonorth_nosplit.params)
    obj_dml_plr_nonorth_nosplit.fit(store_predictions=True,store_models=True)
    this_theta_nonorth_nosplit = obj_dml_plr_nonorth_nosplit.coef[0]
    se_nonorth_nosplit=obj_dml_plr_nonorth_nosplit.se[0]
    se_nonorth_nosplit_v2=var_est_dml(obj_dml_plr_nonorth_nosplit)
    obj_dml_data = DoubleMLData.from_arrays(x, y, d)
    obj_dml_plr = DoubleMLPLR(obj_dml_data,
                              ml_l, ml_m, ml_g,
                              n_folds=2,
                              score='IV-type')
    obj_dml_plr.tune(par_grids, search_mode='grid_search');
    print("ML_full_sample",obj_dml_plr.params)
    obj_dml_plr.fit(store_predictions=True,store_models=True)
    this_theta_dml_plr = obj_dml_plr.coef[0]
    se_orth=obj_dml_plr.se[0]
    se_orth_v2=var_est_dml(obj_dml_plr)
    if ME:
        se_nonorth_nosplit1,se_nonorth_nosplit2=var_est_gee(obj_dml_plr_nonorth_nosplit,using_lasso,w1_x_pred_per_add_cons, xw_nm_m,gee_var,gee_params)
        se_orth1,se_orth2=var_est_gee(obj_dml_plr,using_lasso,w1_x_pred_per_add_cons, xw_nm_m,gee_var,gee_params)
        results=[this_theta_nonorth_nosplit-true_para,this_theta_dml_plr-true_para,se_nonorth_nosplit1,se_orth1,se_nonorth_nosplit2,se_orth2]
    else:
        results=[this_theta_nonorth_nosplit-true_para,this_theta_dml_plr-true_para,se_nonorth_nosplit,se_orth,se_nonorth_nosplit_v2,se_orth_v2]
    return results




