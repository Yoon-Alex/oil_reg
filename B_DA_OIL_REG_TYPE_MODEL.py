#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#===========================================================
# File Name : B_DA_OIL_REG_TYPE_ANL MODEL SCRIPT                            
# Description :                                             
# Date : 2020-11-12                                         
# Writer : Yoon Jun Beom
# Packages :                                                
# Note :                           
#===========================================================

# library import 
import os, sys
import jaydebeapi as jdb
import pandas as pd
import numpy as np
import seaborn as sns
import datetime, time
import joblib

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

def oil_reg_type_anl(conn): 
    sql = """
    SELECT  *
      FROM  NEW_BIGDATA.B_DA_OIL_REG_TYPE_ANL_DS
    """
    
    df = pd.read_sql(sql, conn)
    df =  df[['reg_no', 'app_amt', 'app_cnt', 'all_day', 'member_no', 'weekday', 'weekend',
                'avg_app_amt','avg_oil_amt', 'avg_oil_prc', 'commute', 'gungu_avg_prc',
                'max_reg_use_cnt', 'oil_amt', 'oil_prc_diff', 'use_reg_cnt']]

    return df.drop(["app_amt", "oil_amt", "gungu_avg_prc"], axis = 1)

def debit_retl(conn): 
    sql = """
    SELECT  DBR_REG_NO 
            , H_CODE HCODE 
      FROM  NEW_BIGDATA.DEBIT_RETL
    """

    df = pd.read_sql(sql, conn)
    return df[['reg_no','h_code']]

def bulid_n_load_model(oil_cust_dat):
    ym = datetime.datetime.now().strftime("%Y%m")
    
    # 단골 / 일반
    dangol_cust = oil_cust_dat.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["member_no", "use_reg_cnt", "app_cnt" ,"max_reg_use_cnt"]]
    dangol_cust['max_reg_use_rto'] = dangol_cust['max_reg_use_cnt'] / dangol_cust['app_cnt']
    dangol_cust = dangol_cust.drop(["member_no", "app_cnt"], axis = 1 )

    scaler = MinMaxScaler()

    dangol_cust_md = pd.DataFrame(scaler.fit_transform(dangol_cust), columns = dangol_cust.columns)
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(dangol_cust_md.values)
    y_kmeans = kmeans.predict(dangol_cust_md.values)

    dangol_cust['cluster'] = y_kmeans

    md_dangol_cust = dangol_cust.copy()
    y = md_dangol_cust['cluster'].values
    X = md_dangol_cust.drop('cluster', axis = 1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    rf = RandomForestClassifier(n_estimators= 100)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)
    y_pred = (y_prob[:,1] > 0.9).astype("int")

    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])
    # load
    cm_matrix.to_csv(f"dangol_rf_cm_{ym}.csv", index=False)    
    joblib.dump(rf, f'dangol_rf_{ym}.pkl')
    
    # 출퇴근 / 일상
    com_all_dat = oil_cust_dat.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["commute", "all_day"]]
    com_all_dat['commute_rto'] = com_all_dat['commute'] / com_all_dat['all_day']

    com_all_dat_md = pd.DataFrame(scaler.fit_transform(com_all_dat), columns = com_all_dat.columns)
    km_com_all = KMeans(n_clusters=2)

    km_com_all.fit(com_all_dat_md.values)
    y_kmeans = km_com_all.predict(com_all_dat_md.values)

    com_all_dat_md['cluster'] = y_kmeans
    com_all_dat['cluster'] = y_kmeans

    y = com_all_dat['cluster'].values
    X = com_all_dat.drop('cluster', axis = 1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    rf = RandomForestClassifier(n_estimators= 100)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)
    y_pred = (y_prob[:,1] > 0.9).astype("int")

    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])
    # load
    cm_matrix.to_csv(f"commute_rf_cm_{ym}.csv", index=False)    
    joblib.dump(rf, f'commute_rf_{ym}.pkl')    
    
    # 가득 / 알뜰     
    full_min_dat = oil_cust_dat.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["avg_app_amt", "avg_oil_amt", "oil_prc_diff", "avg_oil_prc"]]

    full_min_dat_md = pd.DataFrame(scaler.fit_transform(full_min_dat), columns = full_min_dat.columns)
    km_full_min = KMeans(n_clusters=2)

    km_full_min.fit(full_min_dat_md.values)
    y_kmeans = km_full_min.predict(full_min_dat_md.values)

    full_min_dat_md['cluster'] = y_kmeans
    full_min_dat['cluster'] = y_kmeans

    y = full_min_dat['cluster'].values
    X = full_min_dat.drop('cluster', axis = 1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    rf = RandomForestClassifier(n_estimators= 100)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)
    y_pred = (y_prob[:,1] > 0.9).astype("int")

    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])
    # load
    cm_matrix.to_csv(f"full_min_rf_cm_{ym}.csv", index=False)    
    joblib.dump(rf, f'full_min_rf_{ym}.pkl')    
    
    # 주중 / 주말
    holy_dat = oil_cust_dat.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["weekday", "weekend"]]
    holy_dat['holy_rto'] = holy_dat['weekend'] / (holy_dat['weekend'] + holy_dat['weekday'])

    holy_dat_md = pd.DataFrame(scaler.fit_transform(holy_dat), columns = holy_dat.columns)
    km_holy = KMeans(n_clusters=2)

    km_holy.fit(holy_dat_md.values)
    y_kmeans = km_holy.predict(holy_dat_md.values)

    holy_dat_md['cluster'] = y_kmeans
    holy_dat['cluster'] = y_kmeans

    y = holy_dat['cluster'].values
    X = holy_dat.drop('cluster', axis = 1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    rf = RandomForestClassifier(n_estimators= 100)
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)
    y_pred = (y_prob[:,1] > 0.9).astype("int")

    cm = confusion_matrix(y_test, y_pred)
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                     index=['Predict Positive:1', 'Predict Negative:0'])
    # load
    cm_matrix.to_csv(f"holy_rf_cm_{ym}.csv", index=False)    
    joblib.dump(rf, f'holy_rf_{ym}.pkl')    
    
if __name__ == "__main__":
    # get parameter
    args_mm = sys.argv[1]

    # Load Data  
    conn = jdb.connect('','',['[id]', '[password]'],'[jar file]')
    
    oil_cust_dat = oil_reg_type_anl(conn)    
    bulid_n_load_model(oil_cust_dat)
    

