#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#===========================================================
# File Name : B_DA_OIL_REG_TYPE_ANL
# Description :                                             
# Date :                                    
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
import time
import joblib

def oil_reg_type_anl(conn): 
    sql = """
    SELECT  *
      FROM  NEW_BIGDATA.B_DA_OIL_REG_TYPE_ANL_DS
    """
    
    df = pd.read_sql(sql, conn)
    return df

def debit_retl(conn): 
    sql = """
    SELECT  DBR_REG_NO 
            , H_CODE HCODE 
      FROM  NEW_BIGDATA.DEBIT_RETL
    """

    df = pd.read_sql(sql, conn)
    return df[['reg_no','h_code']]

def predict(oil_cust_reg_tmp, debit_retl):    
    # --------------- 단골/일반 모델 적용 --------------- #
    member_max_use_cnt = oil_cust_reg_tmp.groupby("member_no", as_index = False).app_cnt.sum()
    member_max_use_cnt.columns = ["member_no", "total_app_cnt"]

    dangol_cust = oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["reg_no", "member_no", "use_reg_cnt", "app_cnt" ,"max_reg_use_cnt"]]
    reg_regul_dat = dangol_cust.merge(member_max_use_cnt, how = 'left', on = "member_no")

    reg_dangol_cust_md = reg_regul_dat.copy()
    reg_dangol_cust_md['max_reg_use_rto'] =     reg_dangol_cust_md['max_reg_use_cnt'] / reg_dangol_cust_md['total_app_cnt']

    reg_dangol_cust_md = reg_dangol_cust_md.drop(["reg_no", "member_no","app_cnt", "total_app_cnt"], axis = 1)
    reg_dangol_cust_md.columns

    model = joblib.load('./model/regular_rf_202011.pkl')

    y_prob = model.predict_proba(reg_dangol_cust_md.values)
    y_pred = (y_prob[:,1] >= 0.9).astype("int")

    dangol_cust['result'] = y_pred

    # -------------- 출퇴근/종일 모델 적용 -------------- #
    com_all_dat = oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80")
                                  .loc[:,["commute", "all_day"]]
    com_all_dat['commute_rto'] = com_all_dat['commute'] / com_all_dat['all_day']

    model = joblib.load('./model/holy_rf_202011.pkl')

    y_prob = model.predict_proba(com_all_dat.values)
    y_pred = (y_prob[:,1] >= 0.9).astype("int")

    com_all_dat['result'] = y_pred

    # --------------- 가득/일반 모델 적용 --------------- #
    full_min_dat = oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["avg_app_amt", "avg_oil_amt", "oil_prc_diff", "avg_oil_prc"]]

    model = joblib.load('./model/full_min_rf_202011.pkl')

    y_prob = model.predict_proba(full_min_dat.values)
    y_pred = (y_prob[:,1] >= 0.9).astype("int")

    full_min_dat['result'] = y_pred

    # ---------------- 주중/주말 모델 적용 -------------- #
    holy_dat = oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80").loc[:,["weekday", "weekend"]]
    holy_dat['holy_rto'] = holy_dat['weekend'] / (holy_dat['weekend'] + holy_dat['weekday'])

    model = joblib.load('./model/holy_rf_202011.pkl')

    y_prob = model.predict_proba(holy_dat.values)
    y_pred = (y_prob[:,1] >= 0.9).astype("int")

    holy_dat['result'] = y_pred
    
    # 단골/일반형 비율 산출
    dancol_cust_r =     dangol_cust.groupby(['reg_no', 'result']).member_no.nunique().reset_index()               .rename(columns = {"result": "regular_pred", "member_no": "regular_pred_cnt"})               .pivot(index= 'reg_no', columns= 'regular_pred', values= 'regular_pred_cnt')               .fillna(0).rename(columns = {0 : "regular_n", 1: "regular_y"})

    dancol_cust_r['regular_rto'] = dancol_cust_r['regular_y']/(dancol_cust_r['regular_n'] + dancol_cust_r['regular_y'])
    dancol_cust_r.replace([np.inf, -np.inf], 0, inplace = True)

    # 출퇴근형/일상형 비율 산출
    com_all_dat_r =     pd.concat([com_all_dat, 
                oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80")
                .loc[:,["reg_no"]]], axis = 1)\
                .groupby(['reg_no', 'result']).commute_rto.count().reset_index()\
                .rename(columns = {"result": "com_all_pred", "commute_rto": "com_all_pred_cnt"})\
                .pivot(index= 'reg_no', columns= 'com_all_pred', values= 'com_all_pred_cnt')\
                .fillna(0).rename(columns = {0 : "all_day", 1: "commute"})

    com_all_dat_r['commute_rto'] = com_all_dat_r['commute']/(com_all_dat_r['commute'] + com_all_dat_r['all_day'])
    com_all_dat_r.replace([np.inf, -np.inf], 0, inplace = True)
    com_all_dat_r.head()

    # 고액/알뜰형 비율 산출
    full_min_dat_r =     pd.concat([full_min_dat, 
                oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80")
                .loc[:,["reg_no"]]], axis = 1)\
                .groupby(['reg_no', 'result']).avg_app_amt.count().reset_index()\
                .rename(columns = {"result": "full_chip_pred", "avg_app_amt": "full_chip_pred_cnt"})\
                .pivot(index= 'reg_no', columns= 'full_chip_pred', values= 'full_chip_pred_cnt')\
                .fillna(0).rename(columns = {0 : "chip", 1: "full"})

    full_min_dat_r['full_rto'] = full_min_dat_r['full']/(full_min_dat_r['full'] + full_min_dat_r['chip'])
    full_min_dat_r.replace([np.inf, -np.inf], 0, inplace = True)
    full_min_dat_r.head()

    # 주중/주말형 비율 산출
    holy_dat_r =     pd.concat([holy_dat, 
                oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80")
                .loc[:,["reg_no"]]], axis = 1)\
                .groupby(['reg_no', 'result']).holy_rto.count().reset_index()\
                .rename(columns = {"result": "weekday_end_pred", "holy_rto": "weekday_end_pred_cnt"})\
                .pivot(index= 'reg_no', columns= 'weekday_end_pred', values= 'weekday_end_pred_cnt')\
                .fillna(0).rename(columns = {0 : "weekday", 1: "weekend"})

    holy_dat_r['week_end_rto'] = holy_dat_r['weekend']/(holy_dat_r['weekend'] + holy_dat_r['weekday'])
    holy_dat_r.replace([np.inf, -np.inf], 0, inplace = True)
    holy_dat_r.head()
    
    # 각 유형 결합
    df = dancol_cust_r.join(com_all_dat_r, how = 'inner')                      .join(full_min_dat_r, how = 'inner')                      .join(holy_dat_r, how = 'inner')
    df = df[['regular_rto','commute_rto','full_rto','week_end_rto']]
    df['regular_std_rto'] = df.median()[0]
    df['commute_std_rto'] = df.median()[1]
    df['full_std_rto'] = df.median()[2]
    df['weekend_std_rto'] = df.median()[3]

    debit_retl['gungu_cd'] = debit_retl.h_code.str.slice(0,5)
    result_df = df.merge(debit_retl[['reg_no', 'gungu_cd']], left_index = True, right_on = 'reg_no')
    oil_cust_reg_tmp.query("app_cnt >=6 & avg_oil_amt >= 8 & avg_oil_amt <= 80")
                                  .loc[:,["commute", "all_day"]]
    
    oil_juyuso_sales =     oil_cust_reg_tmp.groupby("reg_no")                    .agg({"app_cnt":["sum", "mean"] ,"app_amt":"sum",
                          "avg_app_amt":"mean", "avg_oil_amt":"mean",
                          "avg_oil_prc":"mean", "oil_prc_diff":"mean"})

    oil_juyuso_sales.columns = ["app_cnt", "avg_app_cnt", "app_amt", "avg_app_amt", "avg_oil_amt", "avg_oil_prc", "oil_prc_diff"]
    smary_tt_sales =     result_df.merge(oil_juyuso_sales, how = 'left', left_on = 'reg_no', right_index = True)

    smary_tt_sales['md_yearmonth'] = args_mm
    return smary_tt_sales[["md_yearmonth","reg_no", "gungu_cd", 
                            "app_amt", "app_cnt", "avg_app_amt", 
                            "avg_app_cnt", "avg_oil_amt", "avg_oil_prc", 
                            "oil_prc_diff", "commute_rto", "full_rto", 
                            "week_end_rto", "regular_rto","commute_std_rto", 
                            "full_std_rto", "weekend_std_rto", "regular_std_rto"]]

def export(conn, smary_tt_sales):
    cursor = conn.cursor()
    
    #Drop table:
    cursor.execute("DROP TABLE IF EXISTS new_bigdata.tmp_da_oil_reg_anl")
    cursor.close()
    
    smary_tt_sales.to_sql('tmp_da_oil_reg_anl', con = conn, chunksize = 1000, index=False)    
    
if __name__ == "__main__":
    # get parameter
    args_mm = sys.argv[1]

    # Load Data  
    conn = jdb.connect('','',['[id]', '[password]'],'[jar file]')

    oil_cust_reg_tmp = oil_reg_type_anl(conn)
    debit_retl = debit_retl(conn)

    smary_tt_sales = predict(oil_cust_reg_tmp, debit_retl)
    export(conn, smary_tt_sales)    

