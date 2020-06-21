'''
Transform Data
'''
# Import packages
import numpy as np
import pandas as pd
import os.path
import sys
import argparse
#import datasource as ds # to connect to BIM


#find the year migration was made
def cohort_df(df, cpc_list):
    '''
    returns the cpc cohort each row belongs to. cohort 0 = belongs to no cohort
    input: dataframe
    output: dataframe or 1(failed=no cohorts found)
    effects: modifies input df to add new cohort column
    '''

    #iterate through all cpc columns and update cohort
    #Assumes the cpc list comes in ascending order of year. Eg. (cpc201209, cpc201210, etc.)
    for cpc_year in range(1,len(cpc_list)):

        if cpc_year ==1:
            df['cohort'] = np.where(df[cpc_list[cpc_year-1]] != df[cpc_list[cpc_year]], int(cpc_list[cpc_year].strip('cpc')), 0)
        else:
            df['cohort'] = np.where(df[cpc_list[cpc_year-1]] != df[cpc_list[cpc_year]], int(cpc_list[cpc_year].strip('cpc')), df.cohort)
    
    #to save space (int 64 to int 32)
    df.cohort = df.cohort.astype('int32')
    #No Cohorts found
    if not df['cohort'].any():
        print('No Cohort Found')

    return df


def production_flag_cohort(df):
    '''
    Adds tag which identifies accounts to be used for production purposes (Target customers to run the model on) (Non Training or Testing)
    Conditions for prod_tag:
        1) No Migrations in the past year
        2) Has a credit score > 680
        3) Does NOT currently have TAW
        4) Is NOT pending approval for a CC
        5) Is NOT delinquent > 30 Days
    '''

    from datetime import date
    import datetime

    #Uncomment the commented lines below when the data collection is up to date. Currently the most recent cpc date is cpc201908
    #df['cpc_before']=np.where(df['match_cpc_after'].isnull(), df['cpc' + (datetime.datetime.now().strftime('%Y')+datetime.datetime.now().strftime('%m'))], df.cpc_before)

    #Comment the line below when data collection is up to date
    df['cpc_before']=np.where(df['match_cpc_after'].isnull(), df['cpc201908'], df.cpc_before)

    #today_year = np.uint32(date.today().year)
    #today_month = np.uint32(date.today().month)

    #df['cohort_month'] = np.where(df['cohort'] == 0, today_month, df.cohort_month)
    #df['cohort_year'] = np.where(df['cohort'] == 0, today_year, df.cohort_year)
    #df['cohort'] = np.where(df['cohort'] == 0, np.uint32(datetime.datetime.now().strftime('%Y')+datetime.datetime.now().strftime('%m')), df.cohort)
    #df['cohort']=np.where(df['cohort'] == 0, np.uint32(201908), df.cohort)
    #df['cohort_month'] = np.where(df['cohort'] == 0, np.uint32(8), df.cohort_month)
    #df['cohort_year'] = np.where(df['cohort'] == 0, np.uint32(2019), df.cohort_year)

    #check if most recent credit score is low
    #uncomment bottom when data collection is set up to date
    #df['low_credit_score'] = np.where(df['cr_bureau_score'+str(datetime.datetime.now().strftime('%Y')+datetime.datetime.now().strftime('%m'))] < 680, 1, 0)
    df['low_credit_score'] = np.where(df['cr_bureau_score201908'] < 680, 1, 0)

    #check if most recent cpc is TAW
    #uncomment bottom when data collection is set up to date
    #df['non_TAW'] = np.where(df['cpc'+str(datetime.datetime.now().strftime('%Y')+datetime.datetime.now().strftime('%m'))] =='TAW', 1, 0)
    df['non_taw'] = np.where(df['cpc201908'] == 'TAW', 0, 1)

    
    #Add production data tag
    #df['match_prod_tag'] = np.where((df['non_TAW']==1) & (df['low_credit_score']==0) & (df['no_mig']==1) 
    #                                    & (df['delinquent']==1) & (df['pending']==1), 1, 0)

    #Add production data tag
    df['match_prod_tag'] = np.where((df['non_taw']==1) & (df['low_credit_score']==0) & (df['no_mig']==1) 
                                       , 1, 0)


    #remove unnecessary columns
    df = df.drop(['non_taw','low_credit_score'], axis=1)

    return df

    '''
    #Add delinquent tag , delinquent days > 30
    
    with ds.connect('bim') as con:
        sql = """     
    SELECT tsys_acct_id
    into #TMP2
    FROM [BIM_DAILY].[dbo].<TABLE_name>
    """
        #Execute the SQL query (notice that code below is indented to the right as it is written under the 'with' statement)
        con.execute(sql)

        #Read data from the temporary table to a dataframe
        df_delinquent = con.read_table('#TMP2')

    df['delinquent'] = np.where(df['tsys_acct_id'].isin(df_delinquent['tsys_acct_id']), 1, 0)

    #Identify the accounts pending approval for new CC 
    with ds.connect('bim') as con:
        sql = """     
    SELECT tsys_acct_id
    into #TMP1
    FROM [BIM_DAILY].[dbo].[ACQUISITION_DAILY]
	where App_Status = 'PD'
	and tsys_acct_id is not NULL
	and tsys_acct_id != 0
    """
        #Execute the SQL query (notice that code below is indented to the right as it is written under the 'with' statement)
        con.execute(sql)

        #Read data from the temporary table to a dataframe
        df_pending = con.read_table('#TMP1')
    
    df['pending'] = np.where(df['tsys_acct_id'].isin(df_pending['tsys_acct_id']), 1, 0)
    '''


# Define Migration Types of the CC. Eg. TPT-TIC, etc.
def Migration_type(df, cpc_list):
    '''
    returns the migration each row belongs to. migration '' = belongs to no migration
    input: dataframe
    output: dataframe or 1(failed=no migrations found)
    effects: modifies input df to add new migrations column
    '''

    global mig_typ_found 
    mig_typ_found = True

    #iterate through all cpc columns and update migration type
    for cpc_year in range(1,len(cpc_list)):

        if cpc_year ==1:
            df['mig_typ'] = np.where(df[cpc_list[cpc_year-1]] != df[cpc_list[cpc_year]], df[cpc_list[cpc_year-1]]+'-'+df[cpc_list[cpc_year]], '')
        else:
            df['mig_typ'] = np.where(df[cpc_list[cpc_year-1]] != df[cpc_list[cpc_year]], df[cpc_list[cpc_year-1]]+'-'+df[cpc_list[cpc_year]], df.mig_typ)

    #No migrations found
    if not df['mig_typ'].any():
        print('No Migration Types Found')
        mig_typ_found = False

    return df


def cpc_before_after(df, cpc_list):
    '''
    Split the Mig_typ column as cpc_before and cpc_After, and replace MMC to TPB. Drop Mig_typ column
    '''

    if mig_typ_found:
        df[['cpc_before','match_cpc_after']] = df.mig_typ.str.split("-", n=3, expand=True).replace('MMC' , 'TPB')
    else:
        df['cpc_before'] = df[cpc_list[-1]]
        df['match_cpc_after'] = None

    df = df.drop('mig_typ',axis=1)

    return df


def no_migration_flag(df, cpc_list):
    '''
    Adds columns to input dataframe where no cpc migration was made .Example, TIC-TIC. ie, # of unique cpcs per customer = 1
    Input: DataFrame
    Output: Mutated Dataframe
    Effect: Mutates input dataframe
    '''

    df['no_mig'] = np.where(num_unique_per_row(df[cpc_list])==1, 1, 0)

    return df


def remove_no_migration(df):
    '''
    Removes rows from input dataframe where no cpc migration was made & Data is not for production. Example, TIC-TIC
    Input: DataFrame
    Output: Mutated Dataframe
    Effect: Mutates input dataframe
    '''

    df = df[(df['no_mig']==0)|(df['match_prod_tag']==1)]

    df = df.drop('no_mig',axis=1)

    return df


def num_unique_per_row(a):
    '''
    Efficient way to identify the number of unique elements per row in the input dataframe.
    100X more efficient than pandas nunique() for large number of rows.
    Helper function for remove_pure_gamers
    Input: dataframe
    Output: Frequency of unique element pandas dataframe
    '''
    b = np.sort(a,axis=1)
    return pd.DataFrame((b[:,1:] != b[:,:-1]).sum(axis=1)+1)


def remove_pure_gamers(df,cpc_list):
    '''
    Removing customers who move from product A to B to C in the past year. ie, have >2 Unique products in a year
    Input: DataFrame columns should only be cpc_list
    '''

    df['pure_gamers'] = np.where(num_unique_per_row(df[cpc_list])>2, 1, 0)

    #drop pure gamers and the newly created tag
    df = df[df.pure_gamers==0].drop('pure_gamers', axis=1)

    return df


def tag_softgamers(df, cpc_list):
    '''
    Add Tag for customers who move from product A to B back to A in the past year.
    Input: DataFrame,cpc_list
    Note: call this function after adding no_mig tag
    '''

    df['soft_gamers'] = np.where((df[cpc_list[0]]==df[cpc_list[-1]]) & (df.no_mig==0), 1, 0)

    return df


def remove_aeroplan_after(df):
    '''
    Remove rows from input dataframe where the customer is moving into an Aeroplan card
    Input: Dataframe
    Output: Dataframe
    Effect: mutate input dataframe
    '''

    df = df[(df.match_cpc_after!='TAW')|(df.match_cpc_after!='TAC')|(df.match_cpc_after!='TAI')]

    return df


def remove_emerald(df):
    '''
    Remove all rows from input dataframe where customer is moving from or to an Emerald Card
    Input: Dataframe
    Output: Dataframe
    Effect: mutates input dataframe
    '''

    df = df[(df.cpc_before != 'TEF') | (df.cpc_before != 'TEV') | (df.match_cpc_after != 'TEF') | (df.match_cpc_after != 'TEV')]

    return df


def remove_old_cards(df):
    '''
    Remove all rows from input dataframe where customer moved from or to an OLD card. (TDR & TCT)
    '''

    df = df[(df.cpc_before != 'TDR') | (df.cpc_before != 'TCT') | (df.match_cpc_after != 'TDR') | (df.match_cpc_after != 'TCT')]

    return df


def dummy_cpc_before(df):
    '''
    One Hot Encode (Dummify) cpc_before and Drop cpc_before column from inout dataframe
    '''

    df = pd.concat([df, pd.get_dummies(df.cpc_before, prefix='match_cpc_before')], axis=1).drop('cpc_before', axis=1)

    return df
    

def opendate_to_days(df):
    '''
    Adds days since Account Open Date to the input dataframe
    input:dataframe
    output: dataframe
    effect: mutates dataframe
    '''
    from datetime import date
    today = date.today()
    # Adjusted the format 
    Open_Dt = pd.to_datetime(df.acct_open_dt,format ='%d%b%Y:%H:%M:%S')
    # Setting Today with the same date format
    Today = pd.to_datetime(today,format='%Y/%m/%d')
    # Calculate the days passed
    Dys_Pass = Today-Open_Dt
    
    # Merge the pass day with the dataset
    # to_numeric messes up the days:: changed to fix that and also store as int32 memory efficient
    df.loc[:,'match_days_open'] = Dys_Pass.dt.days.astype('int32')

    #Drop unneeded columns
    df = df.drop('acct_open_dt', axis=1)
    return df


def acct_index_pre_move(df):
    '''
    mutate input dataframe to add cols which match chq_ind, loc_ind, employee_ind, sav_ind, tfsa_ind, usd_chq_ind to the cohort
    input: dataframe
    output: dataframe or 1 (no indexes matched)
    effect: mutates input dataframe
    '''

    #init lists with all chq_ind, loc_ind, employee_ind, sav_ind, tfsa_ind, usd_chq_ind  columns from df
    chq_list = list(df.filter(regex='chq_ind').columns)
    loc_list = list(df.filter(regex='loc_ind').columns)
    emp_list = list(df.filter(regex='employee_ind').columns)
    sav_list = list(df.filter(regex='sav_ind').columns)
    tfsa_list = list(df.filter(regex='tfsa_ind').columns)
    usd_chq_list = list(df.filter(regex='usd_chq_ind').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        if cohort_ind == 0:
            df['match_chq_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['chq_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_loc_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['loc_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_employee_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['employee_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_sav_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['sav_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_tfsa_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['tfsa_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_usd_chq_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['usd_chq_ind'+str(cohort_list[cohort_ind])], 0)
        else:
            df['match_chq_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['chq_ind'+str(cohort_list[cohort_ind])], df.match_chq_ind)
            df['match_loc_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['loc_ind'+str(cohort_list[cohort_ind])], df.match_loc_ind)
            df['match_employee_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['employee_ind'+str(cohort_list[cohort_ind])], df.match_employee_ind)
            df['match_sav_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['sav_ind'+str(cohort_list[cohort_ind])], df.match_sav_ind)
            df['match_tfsa_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['tfsa_ind'+str(cohort_list[cohort_ind])], df.match_tfsa_ind)
            df['match_usd_chq_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['usd_chq_ind'+str(cohort_list[cohort_ind])], df.match_usd_chq_ind)

    #set all the new columns to int32 instead of int64 to double efficiency
    df['match_chq_ind'] = df['match_chq_ind'].fillna(0).astype('int32')
    df['match_loc_ind'] = df['match_loc_ind'].fillna(0).astype('int32')
    df['match_employee_ind'] = df['match_employee_ind'].fillna(0).astype('int32')
    df['match_sav_ind'] = df['match_sav_ind'].fillna(0).astype('int32')
    df['match_tfsa_ind'] = df['match_tfsa_ind'].fillna(0).astype('int32')
    df['match_usd_chq_ind'] = df['match_usd_chq_ind'].fillna(0).astype('int32')

    #Remove index columns not required anymore
    df = df.drop(chq_list+loc_list+emp_list+sav_list+tfsa_list+usd_chq_list, axis = 1)
    return df


def match_Amt(df):
    '''
    mutate input dataframe to add cols which match last annual fee amount, last autopayment amt, last change date, last credit limit change to the cohort
    input: dataframe
    output: dataframe or 1 (no last amounts matched)
    effect: mutates input dataframe
    '''
    #init lists with all last annual fee amount, last autopayment amt, last change date, last credit limit change  columns from df
    annfee_list = list(df.filter(regex='last_ann_fee_amt').columns)
    autopay_list = list(df.filter(regex='last_autopayment_amt').columns)
    changedt_list = list(df.filter(regex='last_change_dt').columns)
    credlim_list = list(df.filter(regex='lcrlim_change').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        if cohort_ind == 0:
            df['match_last_ann_fee_amt'] = np.where(df.cohort == cohort_list[cohort_ind], df['last_ann_fee_amt'+str(cohort_list[cohort_ind])], 0)
            df['match_last_autopayment_amt'] = np.where(df.cohort == cohort_list[cohort_ind], df['last_autopayment_amt'+str(cohort_list[cohort_ind])], 0)
            df['match_last_change_dt'] = np.where(df.cohort == cohort_list[cohort_ind], df['last_change_dt'+str(cohort_list[cohort_ind])], 0)
            df['match_lcrlim_change'] = np.where(df.cohort == cohort_list[cohort_ind], df['lcrlim_change'+str(cohort_list[cohort_ind])], 0)
        else:
            df['match_last_ann_fee_amt'] = np.where(df.cohort == cohort_list[cohort_ind], df['last_ann_fee_amt'+str(cohort_list[cohort_ind])], df.match_last_ann_fee_amt)
            df['match_last_autopayment_amt'] = np.where(df.cohort == cohort_list[cohort_ind], df['last_autopayment_amt'+str(cohort_list[cohort_ind])], df.match_last_autopayment_amt)
            df['match_last_change_dt'] = np.where(df.cohort == cohort_list[cohort_ind], df['last_change_dt'+str(cohort_list[cohort_ind])], df.match_last_change_dt)
            df['match_lcrlim_change'] = np.where(df.cohort == cohort_list[cohort_ind], df['lcrlim_change'+str(cohort_list[cohort_ind])], df.match_lcrlim_change)

    #set all the new columns to int32 instead of int64 to double efficiency
    df['match_last_ann_fee_amt'] = df['match_last_ann_fee_amt'].fillna(0).astype('int32')
    df['match_last_autopayment_amt'] = df['match_last_autopayment_amt'].astype('int32')
    df['match_last_change_dt'] = df['match_last_change_dt'].astype('int32')
    df['match_lcrlim_change'] = np.where(df['match_lcrlim_change']==0, df['acct_open_dt'], df['match_lcrlim_change'])

    # Transform date cred limit change to days passed since the day
    from datetime import date
    today = date.today()
    # Setting Today with the same date format
    Today = pd.to_datetime(today,format='%Y/%m/%d')

    lcrlim_change = pd.to_datetime(df.match_lcrlim_change,format ='%d%b%Y:%H:%M:%S')
    # Calculate the pass day
    lcrlim_dys_pass = Today - lcrlim_change

    # to_numeric messes up the days:: changed to fix that and also store as int32 memory efficient
    df.loc[:,'match_lcrlim_dys_pass'] = lcrlim_dys_pass.dt.days.astype('int32')

    # Transform change date to days passed since the day
    Change_dt = pd.to_datetime(df.match_last_change_dt,format ='%Y%M')
    # Calculate the pass day
    Change_Dys_Pass = Today - Change_dt

    # to_numeric messes up the days:: changed to fix that and also store as int32 memory efficient
    df.loc[:,'match_change_date_dys_pass'] = Change_Dys_Pass.dt.days.astype('int32')

    #drop unneeded columns
    df = df.drop(annfee_list+autopay_list+changedt_list+credlim_list+['match_lcrlim_change', 'match_last_change_dt'], axis = 1)

    return df


def avg_three_months_prior_financial(df):
    '''
    Adds features like adb, Net Revenue, nibt, Amt Pymt Applied, Amt Revolved, Credit Limit Amt to the input dataframe. Avg of last 3 months
    prior to migration
    input: dataframe
    output: dataframe
    effect: mutates input dataframe
    '''
    #init lists with all adb, netrev , nibt, Amt Paid, Amt Revolved, Cred limit columns from df
    adb_list = list(df.filter(regex='adb').columns)
    netrev_list = list(df.filter(regex='net_revenue').columns)
    nibt_list = list(df.filter(regex='nibt').columns)
    amtpaid_list = list(df.filter(regex='amt_pymt_applied').columns)
    amtrevolve_list = list(df.filter(regex='amt_revolve').columns)
    credlim_list = list(df.filter(regex='credit_limit_amt').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        #check if cohort in january or february
        january = '01' in str(cohort_list[cohort_ind])[-2:]
        feb = '02' in str(cohort_list[cohort_ind])[-2:]
        march = '03' in str(cohort_list[cohort_ind])[-2:]
        year = str(cohort_list[cohort_ind])[:-2]

        if january:
            if cohort_ind == 0:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'10']+df['adb'+str(int(year)-1)+'12']+df['adb'+str(int(year)-1)+'11'])/3, 0)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'10']+ df['net_revenue'+str(int(year)-1)+'12']+ df['net_revenue'+str(int(year)-1)+'11'])/3, 0)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'10']+df['nibt'+str(int(year)-1)+'12']+df['nibt'+str(int(year)-1)+'11'])/3, 0)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'10']+df['amt_pymt_applied'+str(int(year)-1)+'12']+df['amt_pymt_applied'+str(int(year)-1)+'11'])/3, 0)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'10']+df['amt_revolve'+str(int(year)-1)+'12']+df['amt_revolve'+str(int(year)-1)+'11'])/3, 0)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'10']+df['credit_limit_amt'+str(int(year)-1)+'12']+df['credit_limit_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'10']-df['adb'+str(int(year)-1)+'11']), 0)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'10']-df['net_revenue'+str(int(year)-1)+'11']), 0)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'10']-df['nibt'+str(int(year)-1)+'11']), 0)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'10']-df['amt_pymt_applied'+str(int(year)-1)+'11']), 0)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'10']-df['amt_revolve'+str(int(year)-1)+'11']), 0)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'10']-df['credit_limit_amt'+str(int(year)-1)+'11']), 0)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'11']-df['adb'+str(int(year)-1)+'12']), 0)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'11']-df['net_revenue'+str(int(year)-1)+'12']), 0)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'11']-df['nibt'+str(int(year)-1)+'12']), 0)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'11']-df['amt_pymt_applied'+str(int(year)-1)+'12']), 0)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'11']-df['amt_revolve'+str(int(year)-1)+'12']), 0)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'11']-df['credit_limit_amt'+str(int(year)-1)+'12']), 0)
            else:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'10']+df['adb'+str(int(year)-1)+'12']+df['adb'+str(int(year)-1)+'11'])/3, df.match_adb_prev_3months)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'10']+ df['net_revenue'+str(int(year)-1)+'12']+ df['net_revenue'+str(int(year)-1)+'11'])/3, df.match_netrev_prev_3months)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'10']+df['nibt'+str(int(year)-1)+'12']+df['nibt'+str(int(year)-1)+'11'])/3, df.match_nibt_prev_3months)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'10']+df['amt_pymt_applied'+str(int(year)-1)+'12']+df['amt_pymt_applied'+str(int(year)-1)+'11'])/3, df.match_amtpaid_prev_3months)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'10']+df['amt_revolve'+str(int(year)-1)+'12']+df['amt_revolve'+str(int(year)-1)+'11'])/3, df.match_amtrevolve_prev_3months)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'10']+df['credit_limit_amt'+str(int(year)-1)+'12']+df['credit_limit_amt'+str(int(year)-1)+'11'])/3, df.match_credlimit_prev_3months)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'10']-df['adb'+str(int(year)-1)+'11']), df.match_adb_prev_3months_delta_3_2)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'10']-df['net_revenue'+str(int(year)-1)+'11']), df.match_netrev_prev_3months_delta_3_2)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'10']-df['nibt'+str(int(year)-1)+'11']), df.match_nibt_prev_3months_delta_3_2)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'10']-df['amt_pymt_applied'+str(int(year)-1)+'11']), df.match_amtpaid_prev_3months_delta_3_2)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'10']-df['amt_revolve'+str(int(year)-1)+'11']), df.match_amtrevolve_prev_3months_delta_3_2)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'10']-df['credit_limit_amt'+str(int(year)-1)+'11']), df.match_credlimit_prev_3months_delta_3_2)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'11']-df['adb'+str(int(year)-1)+'12']), df.match_adb_prev_3months_delta_2_1)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'11']-df['net_revenue'+str(int(year)-1)+'12']), df.match_netrev_prev_3months_delta_2_1)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'11']-df['nibt'+str(int(year)-1)+'12']), df.match_nibt_prev_3months_delta_2_1)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'11']-df['amt_pymt_applied'+str(int(year)-1)+'12']), df.match_amtpaid_prev_3months_delta_2_1)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'11']-df['amt_revolve'+str(int(year)-1)+'12']), df.match_amtrevolve_prev_3months_delta_2_1)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'11']-df['credit_limit_amt'+str(int(year)-1)+'12']), df.match_credlimit_prev_3months_delta_2_1)
        elif feb:
            if cohort_ind == 0:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'11']+df['adb'+year+'01']+df['adb'+str(int(year)-1)+'12'])/3, 0)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'11']+ df['net_revenue'+year+'01']+ df['net_revenue'+str(int(year)-1)+'12'])/3, 0)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'11']+df['nibt'+year+'01']+df['nibt'+str(int(year)-1)+'12'])/3, 0)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'11']+df['amt_pymt_applied'+year+'01']+df['amt_pymt_applied'+str(int(year)-1)+'12'])/3, 0)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'11']+df['amt_revolve'+year+'01']+df['amt_revolve'+str(int(year)-1)+'12'])/3, 0)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'11']+df['credit_limit_amt'+year+'01']+df['credit_limit_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'11']-df['adb'+str(int(year)-1)+'12']), 0)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'11']-df['net_revenue'+str(int(year)-1)+'12']), 0)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'11']-df['nibt'+str(int(year)-1)+'12']), 0)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'11']-df['amt_pymt_applied'+str(int(year)-1)+'12']), 0)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'11']-df['amt_revolve'+str(int(year)-1)+'12']), 0)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'11']-df['credit_limit_amt'+str(int(year)-1)+'12']), 0)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'12']-df['adb'+year+'01']), 0)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'12']-df['net_revenue'+year+'01']), 0)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'12']-df['nibt'+year+'01']), 0)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'12']-df['amt_pymt_applied'+year+'01']), 0)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'12']-df['amt_revolve'+year+'01']), 0)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'12']-df['credit_limit_amt'+year+'01']), 0)
                
            else:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'11']+df['adb'+year+'01']+df['adb'+str(int(year)-1)+'12'])/3, df.match_adb_prev_3months)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'11']+ df['net_revenue'+year+'01']+ df['net_revenue'+str(int(year)-1)+'12'])/3, df.match_netrev_prev_3months)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'11']+df['nibt'+year+'01']+df['nibt'+str(int(year)-1)+'12'])/3, df.match_nibt_prev_3months)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'11']+df['amt_pymt_applied'+year+'01']+df['amt_pymt_applied'+str(int(year)-1)+'12'])/3, df.match_amtpaid_prev_3months)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'11']+df['amt_revolve'+year+'01']+df['amt_revolve'+str(int(year)-1)+'12'])/3, df.match_amtrevolve_prev_3months)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'11']+df['credit_limit_amt'+year+'01']+df['credit_limit_amt'+str(int(year)-1)+'12'])/3, df.match_credlimit_prev_3months)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'11']-df['adb'+str(int(year)-1)+'12']), df.match_adb_prev_3months_delta_3_2)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'11']-df['net_revenue'+str(int(year)-1)+'12']), df.match_netrev_prev_3months_delta_3_2)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'11']-df['nibt'+str(int(year)-1)+'12']), df.match_nibt_prev_3months_delta_3_2)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'11']-df['amt_pymt_applied'+str(int(year)-1)+'12']), df.match_amtpaid_prev_3months_delta_3_2)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'11']-df['amt_revolve'+str(int(year)-1)+'12']), df.match_amtrevolve_prev_3months_delta_3_2)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'11']-df['credit_limit_amt'+str(int(year)-1)+'12']), df.match_credlimit_prev_3months_delta_3_2)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'12']-df['adb'+year+'01']), df.match_adb_prev_3months_delta_2_1)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'12']-df['net_revenue'+year+'01']), df.match_netrev_prev_3months_delta_2_1)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'12']-df['nibt'+year+'01']), df.match_nibt_prev_3months_delta_2_1)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'12']-df['amt_pymt_applied'+year+'01']), df.match_amtpaid_prev_3months_delta_2_1)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'12']-df['amt_revolve'+year+'01']), df.match_amtrevolve_prev_3months_delta_2_1)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'12']-df['credit_limit_amt'+year+'01']), df.match_credlimit_prev_3months_delta_2_1)
                
        elif march:
            if cohort_ind == 0:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'02']+df['adb'+year+'01']+df['adb'+str(int(year)-1)+'12'])/3, 0)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+year+'02']+ df['net_revenue'+year+'01']+ df['net_revenue'+str(int(year)-1)+'12'])/3, 0)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+year+'02']+df['nibt'+year+'01']+df['nibt'+str(int(year)-1)+'12'])/3, 0)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+year+'02']+df['amt_pymt_applied'+year+'01']+df['amt_pymt_applied'+str(int(year)-1)+'12'])/3, 0)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+year+'02']+df['amt_revolve'+year+'01']+df['amt_revolve'+str(int(year)-1)+'12'])/3, 0)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+year+'02']+df['credit_limit_amt'+year+'01']+df['credit_limit_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'12']-df['adb'+year+'01']), 0)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'12']-df['net_revenue'+year+'01']), 0)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'12']-df['nibt'+year+'01']), 0)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'12']-df['amt_pymt_applied'+year+'01']), 0)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'12']-df['amt_revolve'+year+'01']), 0)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'12']-df['credit_limit_amt'+year+'01']), 0)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['adb'+year+'02']), 0)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['net_revenue'+year+'02']), 0)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['nibt'+year+'02']), 0)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['amt_pymt_applied'+year+'02']), 0)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['amt_revolve'+year+'02']), 0)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['credit_limit_amt'+year+'02']), 0)
            else:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'02']+df['adb'+year+'01']+df['adb'+str(int(year)-1)+'12'])/3, df.match_adb_prev_3months)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+year+'02']+ df['net_revenue'+year+'01']+ df['net_revenue'+str(int(year)-1)+'12'])/3, df.match_netrev_prev_3months)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+year+'02']+df['nibt'+year+'01']+df['nibt'+str(int(year)-1)+'12'])/3, df.match_nibt_prev_3months)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+year+'02']+df['amt_pymt_applied'+year+'01']+df['amt_pymt_applied'+str(int(year)-1)+'12'])/3, df.match_amtpaid_prev_3months)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+year+'02']+df['amt_revolve'+year+'01']+df['amt_revolve'+str(int(year)-1)+'12'])/3, df.match_amtrevolve_prev_3months)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+year+'02']+df['credit_limit_amt'+year+'01']+df['credit_limit_amt'+str(int(year)-1)+'12'])/3, df.match_credlimit_prev_3months)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(int(year)-1)+'12']-df['adb'+year+'01']), df.match_adb_prev_3months_delta_3_2)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(int(year)-1)+'12']-df['net_revenue'+year+'01']), df.match_netrev_prev_3months_delta_3_2)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(int(year)-1)+'12']-df['nibt'+year+'01']), df.match_nibt_prev_3months_delta_3_2)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(int(year)-1)+'12']-df['amt_pymt_applied'+year+'01']), df.match_amtpaid_prev_3months_delta_3_2)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(int(year)-1)+'12']-df['amt_revolve'+year+'01']), df.match_amtrevolve_prev_3months_delta_3_2)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(int(year)-1)+'12']-df['credit_limit_amt'+year+'01']), df.match_credlimit_prev_3months_delta_3_2)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['adb'+year+'02']), df.match_adb_prev_3months_delta_2_1)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['net_revenue'+year+'02']), df.match_netrev_prev_3months_delta_2_1)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['nibt'+year+'02']), df.match_nibt_prev_3months_delta_2_1)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['amt_pymt_applied'+year+'02']), df.match_amtpaid_prev_3months_delta_2_1)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['amt_revolve'+year+'02']), df.match_amtrevolve_prev_3months_delta_2_1)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+year+'01']-df['credit_limit_amt'+year+'02']), df.match_credlimit_prev_3months_delta_2_1)
        else:
            if cohort_ind == 0:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(cohort_list[cohort_ind]-3)]+df['adb'+str(cohort_list[cohort_ind]-1)]+df['adb'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(cohort_list[cohort_ind]-3)]+ df['net_revenue'+str(cohort_list[cohort_ind]-1)]+ df['net_revenue'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(cohort_list[cohort_ind]-3)]+df['nibt'+str(cohort_list[cohort_ind]-1)]+df['nibt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(cohort_list[cohort_ind]-3)]+df['amt_pymt_applied'+str(cohort_list[cohort_ind]-1)]+df['amt_pymt_applied'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(cohort_list[cohort_ind]-3)]+df['amt_revolve'+str(cohort_list[cohort_ind]-1)]+df['amt_revolve'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(cohort_list[cohort_ind]-3)]+df['credit_limit_amt'+str(cohort_list[cohort_ind]-1)]+df['credit_limit_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(cohort_list[cohort_ind]-3)]-df['adb'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(cohort_list[cohort_ind]-3)]-df['net_revenue'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(cohort_list[cohort_ind]-3)]-df['nibt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(cohort_list[cohort_ind]-3)]-df['amt_pymt_applied'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(cohort_list[cohort_ind]-3)]-df['amt_revolve'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(cohort_list[cohort_ind]-3)]-df['credit_limit_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(cohort_list[cohort_ind]-2)]-df['adb'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(cohort_list[cohort_ind]-2)]-df['net_revenue'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(cohort_list[cohort_ind]-2)]-df['nibt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(cohort_list[cohort_ind]-2)]-df['amt_pymt_applied'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(cohort_list[cohort_ind]-2)]-df['amt_revolve'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(cohort_list[cohort_ind]-2)]-df['credit_limit_amt'+str(cohort_list[cohort_ind]-1)]), 0)
            else:
                df['match_adb_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(cohort_list[cohort_ind]-3)]+df['adb'+str(cohort_list[cohort_ind]-1)]+df['adb'+str(cohort_list[cohort_ind]-2)])/3, df.match_adb_prev_3months)
                df['match_netrev_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(cohort_list[cohort_ind]-3)]+df['net_revenue'+str(cohort_list[cohort_ind]-1)]+df['net_revenue'+str(cohort_list[cohort_ind]-2)])/3, df.match_netrev_prev_3months)
                df['match_nibt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(cohort_list[cohort_ind]-3)]+df['nibt'+str(cohort_list[cohort_ind]-1)]+df['nibt'+str(cohort_list[cohort_ind]-2)])/3, df.match_nibt_prev_3months)
                df['match_amtpaid_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(cohort_list[cohort_ind]-3)]+df['amt_pymt_applied'+str(cohort_list[cohort_ind]-1)]+df['amt_pymt_applied'+str(cohort_list[cohort_ind]-2)])/3, df.match_amtpaid_prev_3months)
                df['match_amtrevolve_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(cohort_list[cohort_ind]-3)]+df['amt_revolve'+str(cohort_list[cohort_ind]-1)]+df['amt_revolve'+str(cohort_list[cohort_ind]-2)])/3, df.match_amtrevolve_prev_3months)
                df['match_credlimit_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(cohort_list[cohort_ind]-3)]+df['credit_limit_amt'+str(cohort_list[cohort_ind]-1)]+df['credit_limit_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_credlimit_prev_3months)
                df['match_adb_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(cohort_list[cohort_ind]-3)]-df['adb'+str(cohort_list[cohort_ind]-2)]), df.match_adb_prev_3months_delta_3_2)
                df['match_netrev_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(cohort_list[cohort_ind]-3)]-df['net_revenue'+str(cohort_list[cohort_ind]-2)]), df.match_netrev_prev_3months_delta_3_2)
                df['match_nibt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(cohort_list[cohort_ind]-3)]-df['nibt'+str(cohort_list[cohort_ind]-2)]), df.match_nibt_prev_3months_delta_3_2)
                df['match_amtpaid_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(cohort_list[cohort_ind]-3)]-df['amt_pymt_applied'+str(cohort_list[cohort_ind]-2)]), df.match_amtpaid_prev_3months_delta_3_2)
                df['match_amtrevolve_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(cohort_list[cohort_ind]-3)]-df['amt_revolve'+str(cohort_list[cohort_ind]-2)]), df.match_amtrevolve_prev_3months_delta_3_2)
                df['match_credlimit_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(cohort_list[cohort_ind]-3)]-df['credit_limit_amt'+str(cohort_list[cohort_ind]-2)]), df.match_credlimit_prev_3months_delta_3_2)
                df['match_adb_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['adb'+str(cohort_list[cohort_ind]-2)]-df['adb'+str(cohort_list[cohort_ind]-1)]), df.match_adb_prev_3months_delta_2_1)
                df['match_netrev_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['net_revenue'+str(cohort_list[cohort_ind]-2)]-df['net_revenue'+str(cohort_list[cohort_ind]-1)]), df.match_netrev_prev_3months_delta_2_1)
                df['match_nibt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nibt'+str(cohort_list[cohort_ind]-2)]-df['nibt'+str(cohort_list[cohort_ind]-1)]), df.match_nibt_prev_3months_delta_2_1)
                df['match_amtpaid_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_pymt_applied'+str(cohort_list[cohort_ind]-2)]-df['amt_pymt_applied'+str(cohort_list[cohort_ind]-1)]), df.match_amtpaid_prev_3months_delta_2_1)
                df['match_amtrevolve_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['amt_revolve'+str(cohort_list[cohort_ind]-2)]-df['amt_revolve'+str(cohort_list[cohort_ind]-1)]), df.match_amtrevolve_prev_3months_delta_2_1)
                df['match_credlimit_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['credit_limit_amt'+str(cohort_list[cohort_ind]-2)]-df['credit_limit_amt'+str(cohort_list[cohort_ind]-1)]), df.match_credlimit_prev_3months_delta_2_1)

    #set all the new columns to int32 instead of int64
    df['match_adb_prev_3months'] = df['match_adb_prev_3months'].astype('int32')
    df['match_netrev_prev_3months'] = df['match_netrev_prev_3months'].astype('int32')
    df['match_nibt_prev_3months'] = df['match_nibt_prev_3months'].astype('int32')
    df['match_amtpaid_prev_3months'] = df['match_amtpaid_prev_3months'].astype('int32')
    df['match_amtrevolve_prev_3months'] = df['match_amtrevolve_prev_3months'].astype('int32')
    df['match_credlimit_prev_3months'] = df['match_credlimit_prev_3months'].astype('int32')
    df['match_adb_prev_3months_delta_3_2'] = df['match_adb_prev_3months_delta_3_2'].astype('float32')
    df['match_netrev_prev_3months_delta_3_2'] = df['match_netrev_prev_3months_delta_3_2'].astype('float32')
    df['match_nibt_prev_3months_delta_3_2'] = df['match_nibt_prev_3months_delta_3_2'].astype('float32')
    df['match_amtpaid_prev_3months_delta_3_2'] = df['match_amtpaid_prev_3months_delta_3_2'].astype('float32')
    df['match_amtrevolve_prev_3months_delta_3_2'] = df['match_amtrevolve_prev_3months_delta_3_2'].astype('float32')
    df['match_credlimit_prev_3months_delta_3_2'] = df['match_credlimit_prev_3months_delta_3_2'].astype('float32')
    df['match_adb_prev_3months_delta_2_1'] = df['match_adb_prev_3months_delta_2_1'].astype('float32')
    df['match_netrev_prev_3months_delta_2_1'] = df['match_netrev_prev_3months_delta_2_1'].astype('float32')
    df['match_nibt_prev_3months_delta_2_1'] = df['match_nibt_prev_3months_delta_2_1'].astype('float32')
    df['match_amtpaid_prev_3months_delta_2_1'] = df['match_amtpaid_prev_3months_delta_2_1'].astype('float32')
    df['match_amtrevolve_prev_3months_delta_2_1'] = df['match_amtrevolve_prev_3months_delta_2_1'].astype('float32')
    df['match_credlimit_prev_3months_delta_2_1'] = df['match_credlimit_prev_3months_delta_2_1'].astype('float32')

    return df

###
# Need to do this for quarters and previous years later
###


def index_customer_basics(df):
    '''
    mutate input dataframe to add cols which match province_code, easyweb_ind, student_ind, secure_acct_ind to the cohort
    input: dataframe
    output: dataframe or 1 (no indexes matched)
    effect: mutates input dataframe
    '''

    #init lists with all chq_ind, loc_ind, employee_ind, sav_ind, usd_chq_ind  columns from df
    #prov_list = list(df.filter(regex='Province_Cd').columns)
    easyweb_list = list(df.filter(regex='easyweb_ind').columns)
    student_list = list(df.filter(regex='student_ind').columns)
    secureacct_list = list(df.filter(regex='secure_acct_ind').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        if cohort_ind == 0:
            #df['match_Province_Cd'] = np.where(df.cohort == cohort_list[cohort_ind], df['Province_Cd'+str(cohort_list[cohort_ind])], 0)
            df['match_easyweb_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['easyweb_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_student_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['student_ind'+str(cohort_list[cohort_ind])], 0)
            df['match_secure_acct_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['secure_acct_ind'+str(cohort_list[cohort_ind])], 0)
        else:
            #df['match_Province_Cd'] = np.where(df.cohort == cohort_list[cohort_ind], df['Province_Cd'+str(cohort_list[cohort_ind])], df.match_Province_Cd)
            df['match_easyweb_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['easyweb_ind'+str(cohort_list[cohort_ind])], df.match_easyweb_ind)
            df['match_student_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['student_ind'+str(cohort_list[cohort_ind])], df.match_student_ind)
            df['match_secure_acct_ind'] = np.where(df.cohort == cohort_list[cohort_ind], df['secure_acct_ind'+str(cohort_list[cohort_ind])], df.match_secure_acct_ind)

    #set all the new columns to int32 instead of int64 to double efficiency
    #df['match_Province_Cd'] = df['match_Province_Cd'].astype('int32')
    df['match_easyweb_ind'] = df['match_easyweb_ind'].fillna(0).astype('int32')
    df['match_student_ind'] = df['match_student_ind'].fillna(0).astype('int32')
    df['match_secure_acct_ind'] = df['match_secure_acct_ind'].fillna(0).astype('int32')

    #Remove index columns not required anymore
    df = df.drop(easyweb_list+student_list+secureacct_list, axis = 1)

    return df


def match_Ind(df):
    '''
    mutate input dataframe to add cols which match EGScore, Credit Score, FIW Flag, SOW Balance, SOW Cl, Triad Align Score, Inactive Months to the cohort
    input: dataframe
    output: dataframe or 1 (no indexes matched)
    effect: mutates input dataframe
    '''

    EG_list = list(df.filter(regex='eg_score').columns)
    CScore_list = list(df.filter(regex='cr_bureau_score').columns)
    FIW_list = list(df.filter(regex='fiw_flg_').columns)
    SOWBal_list = list(df.filter(regex='sow_bal_').columns)
    SOWCl_list = list(df.filter(regex='sow_cl_').columns)
    triad_align_score_list = list(df.filter(regex='triad_align_score').columns)
    inactive_months_list = list(df.filter(regex='inactive_months').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        if cohort_ind == 0:
            df['match_eg_score'] = np.where(df.cohort == cohort_list[cohort_ind], df['eg_score'+str(cohort_list[cohort_ind])], 0)
            df['match_Credit_Score'] = np.where(df.cohort == cohort_list[cohort_ind], df['cr_bureau_score'+str(cohort_list[cohort_ind])], 0)
            df['match_FIW_flg'] = np.where(df.cohort == cohort_list[cohort_ind], df['fiw_flg_'+str(cohort_list[cohort_ind])], 0)
            df['match_SOW_Bal'] = np.where(df.cohort == cohort_list[cohort_ind], df['sow_bal_'+str(cohort_list[cohort_ind])], 0)
            df['match_SOW_CL'] = np.where(df.cohort == cohort_list[cohort_ind], df['sow_cl_'+str(cohort_list[cohort_ind])], 0)
            df['match_triad_align_score'] = np.where(df.cohort == cohort_list[cohort_ind], df['triad_align_score'+str(cohort_list[cohort_ind])], 0)
            df['match_inactive_months'] = np.where(df.cohort == cohort_list[cohort_ind], df['inactive_months'+str(cohort_list[cohort_ind])], 0)
        else:
            df['match_eg_score'] = np.where(df.cohort == cohort_list[cohort_ind], df['eg_score'+str(cohort_list[cohort_ind])], df.match_eg_score)
            df['match_Credit_Score'] = np.where(df.cohort == cohort_list[cohort_ind], df['cr_bureau_score'+str(cohort_list[cohort_ind])], df.match_Credit_Score)
            df['match_FIW_flg'] = np.where(df.cohort == cohort_list[cohort_ind], df['fiw_flg_'+str(cohort_list[cohort_ind])], df.match_FIW_flg)
            df['match_SOW_Bal'] = np.where(df.cohort == cohort_list[cohort_ind], df['sow_bal_'+str(cohort_list[cohort_ind])], df.match_SOW_Bal)
            df['match_SOW_CL'] = np.where(df.cohort == cohort_list[cohort_ind], df['sow_cl_'+str(cohort_list[cohort_ind])], df.match_SOW_CL)
            df['match_triad_align_score'] = np.where(df.cohort == cohort_list[cohort_ind], df['triad_align_score'+str(cohort_list[cohort_ind])], df.match_triad_align_score)
            df['match_inactive_months'] = np.where(df.cohort == cohort_list[cohort_ind], df['inactive_months'+str(cohort_list[cohort_ind])], df.match_inactive_months)

    #set all the new columns to int32 instead of int64 to double efficiency
    df['match_eg_score'] = df['match_eg_score'].fillna(0).astype('int32')
    df['match_Credit_Score'] = df['match_Credit_Score'].fillna(0).astype('int32')
    df['match_FIW_flg'] = df['match_FIW_flg'].astype('int32')
    df['match_SOW_Bal'] = df['match_SOW_Bal'].astype('int32')
    df['match_SOW_CL'] = df['match_SOW_CL'].astype('int32')
    df['match_triad_align_score'] = df['match_triad_align_score'].fillna(0).astype('int32')
    df['match_inactive_months'] = df['match_inactive_months'].astype('int32')


    #Remove index columns not required anymore
    df = df.drop(EG_list+CScore_list+FIW_list+SOWBal_list+SOWCl_list+triad_align_score_list+inactive_months_list, axis = 1)

    return df


def acct_cash_feats(df):
    '''
    Add Cash Adv Cnt, Cash Adv Amt, Cash Bal Amt, Bal Protection fee, NSF Fee, Overlimit Fee columns (Avg 3 months prior move) to input dataframe
    input: Dataframe
    output: Dataframe
    mutates input dataframe
    '''
     #init lists with all Cash Adv Cnt, Cash Adv Amt, Cash Bal Amt, Bal Protection fee, NSF Fee, Overlimit Fee columns from df
    cash_adv_count_list = list(df.filter(regex='cash_advance_count').columns)
    cash_adv_bal_list = list(df.filter(regex='cash_advance_amt').columns)
    Cash_Bal_list = list(df.filter(regex='cash_bal_amt').columns)
    Bal_protect_list = list(df.filter(regex='bal_protection_fee').columns)
    nsf_list = list(df.filter(regex='nsf_fee').columns)
    olimit_fee_list = list(df.filter(regex='overlimit_fee').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        #check if cohort in january or february
        january = '01' in str(cohort_list[cohort_ind])[-2:]
        feb = '02' in str(cohort_list[cohort_ind])[-2:]
        march = '03' in str(cohort_list[cohort_ind])[-2:]
        year = str(cohort_list[cohort_ind])[:-2]

        if january:
            if cohort_ind == 0:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+str(int(year)-1)+'10']+df['cash_advance_count'+str(int(year)-1)+'12']+df['cash_advance_count'+str(int(year)-1)+'11'])/3, 0)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+str(int(year)-1)+'10']+ df['cash_advance_amt'+str(int(year)-1)+'12']+ df['cash_advance_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+str(int(year)-1)+'10']+df['cash_bal_amt'+str(int(year)-1)+'12']+df['cash_bal_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+str(int(year)-1)+'10']+df['bal_protection_fee'+str(int(year)-1)+'12']+df['bal_protection_fee'+str(int(year)-1)+'11'])/3, 0)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+str(int(year)-1)+'10']+df['nsf_fee'+str(int(year)-1)+'12']+df['nsf_fee'+str(int(year)-1)+'11'])/3, 0)
            else:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+str(int(year)-1)+'10']+df['cash_advance_count'+str(int(year)-1)+'12']+df['cash_advance_count'+str(int(year)-1)+'11'])/3, df.match_cash_adv_count_prev_3months)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+str(int(year)-1)+'10']+ df['cash_advance_amt'+str(int(year)-1)+'12']+ df['cash_advance_amt'+str(int(year)-1)+'11'])/3, df.match_cash_adv_bal_prev_3months)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+str(int(year)-1)+'10']+df['cash_bal_amt'+str(int(year)-1)+'12']+df['cash_bal_amt'+str(int(year)-1)+'11'])/3, df.match_bal_protection_fee_prev_3months)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+str(int(year)-1)+'10']+df['bal_protection_fee'+str(int(year)-1)+'12']+df['bal_protection_fee'+str(int(year)-1)+'11'])/3, df.match_nsf_fee_prev_3months)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+str(int(year)-1)+'10']+df['nsf_fee'+str(int(year)-1)+'12']+df['nsf_fee'+str(int(year)-1)+'11'])/3, df.match_overlimit_fee_prev_3months)
        elif feb:
            if cohort_ind == 0:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+str(int(year)-1)+'11']+df['cash_advance_count'+year+'01']+df['cash_advance_count'+str(int(year)-1)+'12'])/3, 0)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+str(int(year)-1)+'11']+ df['cash_advance_amt'+year+'01']+ df['cash_advance_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+str(int(year)-1)+'11']+df['cash_bal_amt'+year+'01']+df['cash_bal_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+str(int(year)-1)+'11']+df['bal_protection_fee'+year+'01']+df['bal_protection_fee'+str(int(year)-1)+'12'])/3, 0)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+str(int(year)-1)+'11']+df['nsf_fee'+year+'01']+df['nsf_fee'+str(int(year)-1)+'12'])/3, 0)
            else:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+str(int(year)-1)+'11']+df['cash_advance_count'+year+'01']+df['cash_advance_count'+str(int(year)-1)+'12'])/3, df.match_cash_adv_count_prev_3months)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+str(int(year)-1)+'11']+ df['cash_advance_amt'+year+'01']+ df['cash_advance_amt'+str(int(year)-1)+'12'])/3, df.match_cash_adv_bal_prev_3months)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+str(int(year)-1)+'11']+df['cash_bal_amt'+year+'01']+df['cash_bal_amt'+str(int(year)-1)+'12'])/3, df.match_bal_protection_fee_prev_3months)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+str(int(year)-1)+'11']+df['bal_protection_fee'+year+'01']+df['bal_protection_fee'+str(int(year)-1)+'12'])/3, df.match_nsf_fee_prev_3months)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+str(int(year)-1)+'11']+df['nsf_fee'+year+'01']+df['nsf_fee'+str(int(year)-1)+'12'])/3, df.match_overlimit_fee_prev_3months)
        elif march:
            if cohort_ind == 0:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+year+'02']+df['cash_advance_count'+year+'01']+df['cash_advance_count'+str(int(year)-1)+'12'])/3, 0)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+year+'02']+ df['cash_advance_amt'+year+'01']+ df['cash_advance_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+year+'02']+df['cash_bal_amt'+year+'01']+df['cash_bal_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+year+'02']+df['bal_protection_fee'+year+'01']+df['bal_protection_fee'+str(int(year)-1)+'12'])/3, 0)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+year+'02']+df['nsf_fee'+year+'01']+df['nsf_fee'+str(int(year)-1)+'12'])/3, 0)
            else:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+year+'02']+df['cash_advance_count'+year+'01']+df['cash_advance_count'+str(int(year)-1)+'12'])/3, df.match_cash_adv_count_prev_3months)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+year+'02']+ df['cash_advance_amt'+year+'01']+ df['cash_advance_amt'+str(int(year)-1)+'12'])/3, df.match_cash_adv_bal_prev_3months)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+year+'02']+df['cash_bal_amt'+year+'01']+df['cash_bal_amt'+str(int(year)-1)+'12'])/3, df.match_bal_protection_fee_prev_3months)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+year+'02']+df['bal_protection_fee'+year+'01']+df['bal_protection_fee'+str(int(year)-1)+'12'])/3, df.match_nsf_fee_prev_3months)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+year+'02']+df['nsf_fee'+year+'01']+df['nsf_fee'+str(int(year)-1)+'12'])/3, df.match_overlimit_fee_prev_3months)
        else:
            if cohort_ind == 0:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+str(cohort_list[cohort_ind]-3)]+df['cash_advance_count'+str(cohort_list[cohort_ind]-1)]+df['cash_advance_count'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+str(cohort_list[cohort_ind]-3)]+ df['cash_advance_amt'+str(cohort_list[cohort_ind]-1)]+ df['cash_advance_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+str(cohort_list[cohort_ind]-3)]+df['cash_bal_amt'+str(cohort_list[cohort_ind]-1)]+df['cash_bal_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+str(cohort_list[cohort_ind]-3)]+df['bal_protection_fee'+str(cohort_list[cohort_ind]-1)]+df['bal_protection_fee'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+str(cohort_list[cohort_ind]-3)]+df['nsf_fee'+str(cohort_list[cohort_ind]-1)]+df['nsf_fee'+str(cohort_list[cohort_ind]-2)])/3, 0)
            else:
                df['match_cash_adv_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_count'+str(cohort_list[cohort_ind]-3)]+df['cash_advance_count'+str(cohort_list[cohort_ind]-1)]+df['cash_advance_count'+str(cohort_list[cohort_ind]-2)])/3, df.match_cash_adv_count_prev_3months)
                df['match_cash_adv_bal_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_advance_amt'+str(cohort_list[cohort_ind]-3)]+df['cash_advance_amt'+str(cohort_list[cohort_ind]-1)]+df['cash_advance_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_cash_adv_bal_prev_3months)
                df['match_bal_protection_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cash_bal_amt'+str(cohort_list[cohort_ind]-3)]+df['cash_bal_amt'+str(cohort_list[cohort_ind]-1)]+df['cash_bal_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_bal_protection_fee_prev_3months)
                df['match_nsf_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['bal_protection_fee'+str(cohort_list[cohort_ind]-3)]+df['bal_protection_fee'+str(cohort_list[cohort_ind]-1)]+df['bal_protection_fee'+str(cohort_list[cohort_ind]-2)])/3, df.match_nsf_fee_prev_3months)
                df['match_overlimit_fee_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['nsf_fee'+str(cohort_list[cohort_ind]-3)]+df['nsf_fee'+str(cohort_list[cohort_ind]-1)]+df['nsf_fee'+str(cohort_list[cohort_ind]-2)])/3, df.match_overlimit_fee_prev_3months)

    #set all the new columns to int32 instead of int64
    df['match_cash_adv_count_prev_3months'] = df['match_cash_adv_count_prev_3months'].astype('int32')
    df['match_cash_adv_bal_prev_3months'] = df['match_cash_adv_bal_prev_3months'].astype('int32')
    df['match_bal_protection_fee_prev_3months'] = df['match_bal_protection_fee_prev_3months'].astype('int32')
    df['match_nsf_fee_prev_3months'] = df['match_nsf_fee_prev_3months'].astype('int32')
    df['match_overlimit_fee_prev_3months'] = df['match_overlimit_fee_prev_3months'].astype('int32')

    #Remove index columns not required anymore
    df = df.drop(cash_adv_count_list+cash_adv_bal_list+Cash_Bal_list+Bal_protect_list+nsf_list+olimit_fee_list, axis = 1)
    
    return df


def rewards_acct(df):
    '''
    Add Apple Pay Amount, Apple pay count , rewards accrued, aeroplan expense amt columns to input dataframe (Avg 3 months prior move)
    input: dataframe
    output: dataframe
    effect: mutates input dataframe
    '''
     #init lists with all Apple Pay Amount, Apple pay count , rewards accrued, aeroplan expense amt columns from df
    aero_amt_list = list(df.filter(regex='aeroplan_expense_amt').columns)
    apple_pay_amt_list = list(df.filter(regex='apple_pay_purchase_amt').columns)
    apple_pay_count_list = list(df.filter(regex='apple_pay_purchase_cnt').columns)
    rewards_list = list(df.filter(regex='rewards_accrued').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        #check if cohort in january or february
        january = '01' in str(cohort_list[cohort_ind])[-2:]
        feb = '02' in str(cohort_list[cohort_ind])[-2:]
        march = '03' in str(cohort_list[cohort_ind])[-2:]
        year = str(cohort_list[cohort_ind])[:-2]

        if january:
            if cohort_ind == 0:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'10']+df['aeroplan_expense_amt'+str(int(year)-1)+'12']+df['aeroplan_expense_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'10']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'12']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'10']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'11'])/3, 0)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'10']+df['rewards_accrued'+str(int(year)-1)+'12']+df['rewards_accrued'+str(int(year)-1)+'11'])/3, 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'10']-df['aeroplan_expense_amt'+str(int(year)-1)+'11']), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'10']-df['apple_pay_purchase_amt'+str(int(year)-1)+'11']), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'10']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']), 0)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'10']-df['rewards_accrued'+str(int(year)-1)+'11']), 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'11']-df['aeroplan_expense_amt'+str(int(year)-1)+'12']), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'11']-df['apple_pay_purchase_amt'+str(int(year)-1)+'12']), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']), 0)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'11']-df['rewards_accrued'+str(int(year)-1)+'12']), 0)
            else:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'10']+df['aeroplan_expense_amt'+str(int(year)-1)+'12']+df['aeroplan_expense_amt'+str(int(year)-1)+'11'])/3, df.match_aeroplan_expense_amount_prev_3months)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'10']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'12']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'11'])/3, df.match_apple_pay_purchase_amount_prev_3months)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'10']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'11'])/3, df.match_apple_pay_purchase_count_prev_3months)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'10']+df['rewards_accrued'+str(int(year)-1)+'12']+df['rewards_accrued'+str(int(year)-1)+'11'])/3, df.match_rewards_accrued_3months)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'10']-df['aeroplan_expense_amt'+str(int(year)-1)+'11']), df.match_aeroplan_expense_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'10']-df['apple_pay_purchase_amt'+str(int(year)-1)+'11']), df.match_apple_pay_purchase_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'10']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']), df.match_apple_pay_purchase_count_prev_3months_delta_3_2)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'10']-df['rewards_accrued'+str(int(year)-1)+'11']), df.match_rewards_accrued_3months_delta_3_2)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'11']-df['aeroplan_expense_amt'+str(int(year)-1)+'12']), df.match_aeroplan_expense_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'11']-df['apple_pay_purchase_amt'+str(int(year)-1)+'12']), df.match_apple_pay_purchase_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']), df.match_apple_pay_purchase_count_prev_3months_delta_2_1)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'11']-df['rewards_accrued'+str(int(year)-1)+'12']), df.match_rewards_accrued_3months_delta_2_1)
        elif feb:
            if cohort_ind == 0:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'11']+df['aeroplan_expense_amt'+year+'01']+df['aeroplan_expense_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'11']+ df['apple_pay_purchase_amt'+year+'01']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']+df['apple_pay_purchase_cnt'+year+'01']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'12'])/3, 0)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+year+'01']+df['rewards_accrued'+str(int(year)-1)+'11']+df['rewards_accrued'+str(int(year)-1)+'12'])/3, 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'11']-df['aeroplan_expense_amt'+str(int(year)-1)+'12']), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'11']-df['apple_pay_purchase_amt'+str(int(year)-1)+'12']), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']), 0)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'11']-df['rewards_accrued'+str(int(year)-1)+'12']), 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'12']-df['aeroplan_expense_amt'+year+'01']), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'12']-df['apple_pay_purchase_amt'+year+'01']), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']-df['apple_pay_purchase_cnt'+year+'01']), 0)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'12']-df['rewards_accrued'+year+'01']), 0)
            else:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'11']+df['aeroplan_expense_amt'+year+'01']+df['aeroplan_expense_amt'+str(int(year)-1)+'12'])/3, df.match_aeroplan_expense_amount_prev_3months)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'11']+ df['apple_pay_purchase_amt'+year+'01']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'12'])/3, df.match_apple_pay_purchase_amount_prev_3months)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']+df['apple_pay_purchase_cnt'+year+'01']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'12'])/3, df.match_apple_pay_purchase_count_prev_3months)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+year+'01']+df['rewards_accrued'+str(int(year)-1)+'11']+df['rewards_accrued'+str(int(year)-1)+'12'])/3, df.match_rewards_accrued_3months)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'11']-df['aeroplan_expense_amt'+str(int(year)-1)+'12']), df.match_aeroplan_expense_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'11']-df['apple_pay_purchase_amt'+str(int(year)-1)+'12']), df.match_apple_pay_purchase_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'11']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']), df.match_apple_pay_purchase_count_prev_3months_delta_3_2)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'11']-df['rewards_accrued'+str(int(year)-1)+'12']), df.match_rewards_accrued_3months_delta_3_2)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(int(year)-1)+'12']-df['aeroplan_expense_amt'+year+'01']), df.match_aeroplan_expense_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(int(year)-1)+'12']-df['apple_pay_purchase_amt'+year+'01']), df.match_apple_pay_purchase_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']-df['apple_pay_purchase_cnt'+year+'01']), df.match_apple_pay_purchase_count_prev_3months_delta_2_1)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(int(year)-1)+'12']-df['rewards_accrued'+year+'01']), df.match_rewards_accrued_3months_delta_2_1)
        elif march:
            if cohort_ind == 0:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+year+'02']+df['aeroplan_expense_amt'+year+'01']+df['aeroplan_expense_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+year+'02']+ df['apple_pay_purchase_amt'+year+'01']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'12'])/3, 0)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+year+'02']+df['apple_pay_purchase_cnt'+year+'01']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'12'])/3, 0)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+year+'02']+df['rewards_accrued'+year+'01']+df['rewards_accrued'+str(int(year)-1)+'12'])/3, 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['aeroplan_expense_amt'+year+'01']-df['aeroplan_expense_amt'+str(int(year)-1)+'12']), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['apple_pay_purchase_amt'+year+'01']- df['apple_pay_purchase_amt'+str(int(year)-1)+'12']), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['apple_pay_purchase_cnt'+year+'01']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']), 0)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['rewards_accrued'+year+'01']-df['rewards_accrued'+str(int(year)-1)+'12']), 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+year+'01']-df['aeroplan_expense_amt'+year+'02']), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+year+'01']- df['apple_pay_purchase_amt'+year+'02']), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+year+'01']-df['apple_pay_purchase_cnt'+year+'02']), 0)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+year+'01']-df['rewards_accrued'+year+'02']), 0)
            else:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+year+'02']+df['aeroplan_expense_amt'+year+'01']+df['aeroplan_expense_amt'+str(int(year)-1)+'12'])/3, df.match_aeroplan_expense_amount_prev_3months)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+year+'02']+ df['apple_pay_purchase_amt'+year+'01']+ df['apple_pay_purchase_amt'+str(int(year)-1)+'12'])/3, df.match_apple_pay_purchase_amount_prev_3months)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+year+'02']+df['apple_pay_purchase_cnt'+year+'01']+df['apple_pay_purchase_cnt'+str(int(year)-1)+'12'])/3, df.match_apple_pay_purchase_count_prev_3months)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+year+'02']+df['rewards_accrued'+year+'01']+df['rewards_accrued'+str(int(year)-1)+'12'])/3, df.match_rewards_accrued_3months)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['aeroplan_expense_amt'+year+'01']-df['aeroplan_expense_amt'+str(int(year)-1)+'12']), df.match_aeroplan_expense_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['apple_pay_purchase_amt'+year+'01']- df['apple_pay_purchase_amt'+str(int(year)-1)+'12']), df.match_apple_pay_purchase_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['apple_pay_purchase_cnt'+year+'01']-df['apple_pay_purchase_cnt'+str(int(year)-1)+'12']), df.match_apple_pay_purchase_count_prev_3months_delta_3_2)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['rewards_accrued'+year+'01']-df['rewards_accrued'+str(int(year)-1)+'12']), df.match_rewards_accrued_3months_delta_3_2)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+year+'01']-df['aeroplan_expense_amt'+year+'02']), df.match_aeroplan_expense_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+year+'01']- df['apple_pay_purchase_amt'+year+'02']), df.match_apple_pay_purchase_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+year+'01']-df['apple_pay_purchase_cnt'+year+'02']), df.match_apple_pay_purchase_count_prev_3months_delta_2_1)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+year+'01']-df['rewards_accrued'+year+'02']), df.match_rewards_accrued_3months_delta_2_1)
        else:
            if cohort_ind == 0:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-3)]+df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-1)]+df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-3)]+ df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-1)]+ df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-3)]+df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-1)]+df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(cohort_list[cohort_ind]-3)]+df['rewards_accrued'+str(cohort_list[cohort_ind]-1)]+df['rewards_accrued'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-3)]-df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-3)]-df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-3)]-df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(cohort_list[cohort_ind]-3)]-df['rewards_accrued'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-2)]-df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-2)]-df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-2)]-df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(cohort_list[cohort_ind]-2)]-df['rewards_accrued'+str(cohort_list[cohort_ind]-1)]), 0)
            else:
                df['match_aeroplan_expense_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-3)]+df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-1)]+df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_aeroplan_expense_amount_prev_3months)
                df['match_apple_pay_purchase_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-3)]+df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-1)]+df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_apple_pay_purchase_amount_prev_3months)
                df['match_apple_pay_purchase_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-3)]+df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-1)]+df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-2)])/3, df.match_apple_pay_purchase_count_prev_3months)
                df['match_rewards_accrued_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(cohort_list[cohort_ind]-3)]+df['rewards_accrued'+str(cohort_list[cohort_ind]-1)]+df['rewards_accrued'+str(cohort_list[cohort_ind]-2)])/3, df.match_rewards_accrued_3months)
                df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-3)]-df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-2)]), df.match_aeroplan_expense_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-3)]-df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-2)]), df.match_apple_pay_purchase_amount_prev_3months_delta_3_2)
                df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-3)]-df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-2)]), df.match_apple_pay_purchase_count_prev_3months_delta_3_2)
                df['match_rewards_accrued_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(cohort_list[cohort_ind]-3)]-df['rewards_accrued'+str(cohort_list[cohort_ind]-2)]), df.match_rewards_accrued_3months_delta_3_2)
                df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-2)]-df['aeroplan_expense_amt'+str(cohort_list[cohort_ind]-1)]), df.match_aeroplan_expense_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-2)]-df['apple_pay_purchase_amt'+str(cohort_list[cohort_ind]-1)]), df.match_apple_pay_purchase_amount_prev_3months_delta_2_1)
                df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-2)]-df['apple_pay_purchase_cnt'+str(cohort_list[cohort_ind]-1)]), df.match_apple_pay_purchase_count_prev_3months_delta_2_1)
                df['match_rewards_accrued_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['rewards_accrued'+str(cohort_list[cohort_ind]-2)]-df['rewards_accrued'+str(cohort_list[cohort_ind]-1)]), df.match_rewards_accrued_3months_delta_2_1)

    #set all the new columns to int32 instead of int64
    df['match_aeroplan_expense_amount_prev_3months'] = df['match_aeroplan_expense_amount_prev_3months'].astype('int32')
    df['match_apple_pay_purchase_amount_prev_3months'] = df['match_apple_pay_purchase_amount_prev_3months'].astype('int32')
    df['match_apple_pay_purchase_count_prev_3months'] = df['match_apple_pay_purchase_count_prev_3months'].astype('int32')
    df['match_rewards_accrued_3months'] = df['match_rewards_accrued_3months'].astype('int32')
    df['match_aeroplan_expense_amount_prev_3months_delta_3_2'] = df['match_aeroplan_expense_amount_prev_3months_delta_3_2'].astype('float32')
    df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'] = df['match_apple_pay_purchase_amount_prev_3months_delta_3_2'].astype('float32')
    df['match_apple_pay_purchase_count_prev_3months_delta_3_2'] = df['match_apple_pay_purchase_count_prev_3months_delta_3_2'].astype('float32')
    df['match_rewards_accrued_3months_delta_3_2'] = df['match_rewards_accrued_3months_delta_3_2'].astype('float32')
    df['match_aeroplan_expense_amount_prev_3months_delta_2_1'] = df['match_aeroplan_expense_amount_prev_3months_delta_2_1'].astype('float32')
    df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'] = df['match_apple_pay_purchase_amount_prev_3months_delta_2_1'].astype('float32')
    df['match_apple_pay_purchase_count_prev_3months_delta_2_1'] = df['match_apple_pay_purchase_count_prev_3months_delta_2_1'].astype('float32')
    df['match_rewards_accrued_3months_delta_2_1'] = df['match_rewards_accrued_3months_delta_2_1'].astype('float32')
    
    #Remove index columns not required anymore
    df = df.drop(aero_amt_list+apple_pay_amt_list+apple_pay_count_list+rewards_list, axis = 1)
    
    return df
    

def transaction_acct(df):
    '''
    Add Count net Transactions, Net Transaction Amount, Travel Count, Travel Amt, Need Count, Need Amt, Want Cnt, Want Amt,
    interchange, Pre Auth Cnt and Pre Auth Amt (Avg 3 months prior to move) to input dataframe

    Input: Dataframe
    Output: Dataframe
    Effect: mutates input dataframe
    '''
     #init lists with columns from df
    cnt_net_trans_list = list(df.filter(regex='cnt_net_transactions').columns)
    net_trans_amt_list = list(df.filter(regex='sum_net_transactions_amt').columns)
    trav_count_list = list(df.filter(regex='travel_cnt').columns)
    trav_amt_list = list(df.filter(regex='travel_amt').columns)
    need_cnt_list = list(df.filter(regex='need_cnt').columns)
    need_amt_list = list(df.filter(regex='need_amt').columns)
    want_count_list = list(df.filter(regex='want_cnt').columns)
    want_amt_list = list(df.filter(regex='want_amt').columns)
    interchange_list = list(df.filter(regex='interchange').columns)
    preauth_amt_count_list = list(df.filter(regex='preauthorized_payment_amt').columns)
    preauth_cnt_list = list(df.filter(regex='preauthorized_payment_cnt').columns)

    #init list of all unique cohorts
    cohort_list = list(np.sort(df.cohort.unique()))

    #iterate through all possible cohorts and create new cols for match index given the cohort matches
    for cohort_ind in range(len(cohort_list)):
        #check if cohort in january or february
        january = '01' in str(cohort_list[cohort_ind])[-2:]
        feb = '02' in str(cohort_list[cohort_ind])[-2:]
        march = '03' in str(cohort_list[cohort_ind])[-2:]
        year = str(cohort_list[cohort_ind])[:-2]

        if january:
            if cohort_ind == 0:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'10']+df['cnt_net_transactions'+str(int(year)-1)+'12']+df['cnt_net_transactions'+str(int(year)-1)+'11'])/3, 0)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'10']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['sum_net_transactions_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'10']+df['travel_cnt'+str(int(year)-1)+'12']+df['travel_cnt'+str(int(year)-1)+'11'])/3, 0)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'10']+df['travel_amt'+str(int(year)-1)+'12']+df['travel_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'10']+df['need_cnt'+str(int(year)-1)+'12']+df['need_cnt'+str(int(year)-1)+'11'])/3, 0)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'10']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['need_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'10']+df['want_cnt'+str(int(year)-1)+'12']+df['want_cnt'+str(int(year)-1)+'11'])/3, 0)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'10']+df['want_amt'+str(int(year)-1)+'12']+df['want_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'10']+ df['interchange'+str(int(year)-1)+'12']+ df['interchange'+str(int(year)-1)+'11'])/3, 0)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'10']+df['preauthorized_payment_amt'+str(int(year)-1)+'12']+df['preauthorized_payment_amt'+str(int(year)-1)+'11'])/3, 0)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'10']+df['preauthorized_payment_cnt'+str(int(year)-1)+'12']+df['preauthorized_payment_cnt'+str(int(year)-1)+'11'])/3, 0)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'10']-df['cnt_net_transactions'+str(int(year)-1)+'11']), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'10']-df['sum_net_transactions_amt'+str(int(year)-1)+'11']), 0)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'10']-df['travel_cnt'+str(int(year)-1)+'11']), 0)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'10']-df['travel_amt'+str(int(year)-1)+'11']), 0)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'10']-df['need_cnt'+str(int(year)-1)+'11']), 0)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'10']-df['need_amt'+str(int(year)-1)+'11']), 0)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'10']-df['want_cnt'+str(int(year)-1)+'11']), 0)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'10']-df['want_amt'+str(int(year)-1)+'11']), 0)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'10']-df['interchange'+str(int(year)-1)+'11']), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'10']-df['preauthorized_payment_amt'+str(int(year)-1)+'11']), 0)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'10']-df['preauthorized_payment_cnt'+str(int(year)-1)+'11']), 0)
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'11']-df['cnt_net_transactions'+str(int(year)-1)+'12']), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'11']-df['sum_net_transactions_amt'+str(int(year)-1)+'12']), 0)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'11']-df['travel_cnt'+str(int(year)-1)+'12']), 0)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'11']-df['travel_amt'+str(int(year)-1)+'12']), 0)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'11']-df['need_cnt'+str(int(year)-1)+'12']), 0)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'11']-df['need_amt'+str(int(year)-1)+'12']), 0)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'11']-df['want_cnt'+str(int(year)-1)+'12']), 0)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'11']-df['want_amt'+str(int(year)-1)+'12']), 0)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'11']-df['interchange'+str(int(year)-1)+'12']), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'11']-df['preauthorized_payment_amt'+str(int(year)-1)+'12']), 0)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'11']-df['preauthorized_payment_cnt'+str(int(year)-1)+'12']), 0)

            else:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'10']+df['cnt_net_transactions'+str(int(year)-1)+'12']+df['cnt_net_transactions'+str(int(year)-1)+'11'])/3, df.match_count_net_transactions_prev_3months)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'10']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['sum_net_transactions_amt'+str(int(year)-1)+'11'])/3, df.match_sum_net_transactions_amt_prev_3months)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'10']+df['travel_cnt'+str(int(year)-1)+'12']+df['travel_cnt'+str(int(year)-1)+'11'])/3, df.match_travel_count_prev_3months)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'10']+df['travel_amt'+str(int(year)-1)+'12']+df['travel_amt'+str(int(year)-1)+'11'])/3, df.match_travel_amount_3months)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'10']+ df['need_cnt'+str(int(year)-1)+'12']+ df['need_cnt'+str(int(year)-1)+'11'])/3, df.match_need_count_prev_3months)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'10']+df['need_amt'+str(int(year)-1)+'12']+df['need_amt'+str(int(year)-1)+'11'])/3, df.match_need_amount_prev_3months)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'10']+df['want_cnt'+str(int(year)-1)+'12']+df['want_cnt'+str(int(year)-1)+'11'])/3, df.match_want_count_prev_3months)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'10']+df['want_amt'+str(int(year)-1)+'12']+df['want_amt'+str(int(year)-1)+'11'])/3, df.match_want_amount_3months)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'10']+ df['interchange'+str(int(year)-1)+'12']+ df['interchange'+str(int(year)-1)+'11'])/3, df.match_interchange_prev_3months)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'10']+df['preauthorized_payment_amt'+str(int(year)-1)+'12']+df['preauthorized_payment_amt'+str(int(year)-1)+'11'])/3, df.match_preauthorized_payment_amt_prev_3months)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'10']+df['preauthorized_payment_cnt'+str(int(year)-1)+'12']+df['preauthorized_payment_cnt'+str(int(year)-1)+'11'])/3, df.match_preauthorized_payment_cnt_3months)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'10']-df['cnt_net_transactions'+str(int(year)-1)+'11']), df.match_count_net_transactions_prev_3months_delta_3_2)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'10']-df['sum_net_transactions_amt'+str(int(year)-1)+'11']), df.match_sum_net_transactions_amt_prev_3months_delta_3_2)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'10']-df['travel_cnt'+str(int(year)-1)+'11']), df.match_travel_count_prev_3months_delta_3_2)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'10']-df['travel_amt'+str(int(year)-1)+'11']), df.match_travel_amount_3months_delta_3_2)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'10']-df['need_cnt'+str(int(year)-1)+'11']), df.match_need_count_prev_3months_delta_3_2)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'10']-df['need_amt'+str(int(year)-1)+'11']), df.match_need_amount_prev_3months_delta_3_2)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'10']-df['want_cnt'+str(int(year)-1)+'11']), df.match_want_count_prev_3months_delta_3_2)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'10']-df['want_amt'+str(int(year)-1)+'11']), df.match_want_amount_3months_delta_3_2)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'10']-df['interchange'+str(int(year)-1)+'11']), df.match_interchange_prev_3months_delta_3_2)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'10']-df['preauthorized_payment_amt'+str(int(year)-1)+'11']), df.match_preauthorized_payment_amt_prev_3months_delta_3_2)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'10']-df['preauthorized_payment_cnt'+str(int(year)-1)+'11']), df.match_preauthorized_payment_cnt_3months_delta_3_2)
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'11']-df['cnt_net_transactions'+str(int(year)-1)+'12']), df.match_count_net_transactions_prev_3months_delta_2_1)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'11']-df['sum_net_transactions_amt'+str(int(year)-1)+'12']), df.match_sum_net_transactions_amt_prev_3months_delta_2_1)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'11']-df['travel_cnt'+str(int(year)-1)+'12']), df.match_travel_count_prev_3months_delta_2_1)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'11']-df['travel_amt'+str(int(year)-1)+'12']), df.match_travel_amount_3months_delta_2_1)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'11']-df['need_cnt'+str(int(year)-1)+'12']), df.match_need_count_prev_3months_delta_2_1)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'11']-df['need_amt'+str(int(year)-1)+'12']), df.match_need_amount_prev_3months_delta_2_1)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'11']-df['want_cnt'+str(int(year)-1)+'12']), df.match_want_count_prev_3months_delta_2_1)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'11']-df['want_amt'+str(int(year)-1)+'12']), df.match_want_amount_3months_delta_2_1)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'11']-df['interchange'+str(int(year)-1)+'12']), df.match_interchange_prev_3months_delta_2_1)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'11']-df['preauthorized_payment_amt'+str(int(year)-1)+'12']), df.match_preauthorized_payment_amt_prev_3months_delta_2_1)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'11']-df['preauthorized_payment_cnt'+str(int(year)-1)+'12']), df.match_preauthorized_payment_cnt_3months_delta_2_1)

        elif feb:
            if cohort_ind == 0:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'11']+df['cnt_net_transactions'+str(int(year)-1)+'12']+df['cnt_net_transactions'+year+'01'])/3, 0)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'11']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['sum_net_transactions_amt'+year+'01'])/3, 0)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'11']+df['travel_cnt'+str(int(year)-1)+'12']+df['travel_cnt'+year+'01'])/3, 0)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'11']+df['travel_amt'+str(int(year)-1)+'12']+df['travel_amt'+year+'01'])/3, 0)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'11']+df['need_cnt'+str(int(year)-1)+'12']+df['need_cnt'+year+'01'])/3, 0)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'11']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['need_amt'+year+'01'])/3, 0)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'11']+df['want_cnt'+str(int(year)-1)+'12']+df['want_cnt'+year+'01'])/3, 0)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'11']+df['want_amt'+str(int(year)-1)+'12']+df['want_amt'+year+'01'])/3, 0)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'11']+ df['interchange'+str(int(year)-1)+'12']+ df['interchange'+year+'01'])/3, 0)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'11']+df['preauthorized_payment_amt'+str(int(year)-1)+'12']+df['preauthorized_payment_amt'+year+'01'])/3, 0)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'11']+df['preauthorized_payment_cnt'+str(int(year)-1)+'12']+df['preauthorized_payment_cnt'+year+'01'])/3, 0)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'11']-df['cnt_net_transactions'+str(int(year)-1)+'12']), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'11']-df['sum_net_transactions_amt'+str(int(year)-1)+'12']), 0)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'11']-df['travel_cnt'+str(int(year)-1)+'12']), 0)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'11']-df['travel_amt'+str(int(year)-1)+'12']), 0)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'11']-df['need_cnt'+str(int(year)-1)+'12']), 0)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'11']-df['sum_net_transactions_amt'+str(int(year)-1)+'12']), 0)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'11']-df['want_cnt'+str(int(year)-1)+'12']), 0)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'11']-df['want_amt'+str(int(year)-1)+'12']), 0)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'11']- df['interchange'+str(int(year)-1)+'12']), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'11']-df['preauthorized_payment_amt'+str(int(year)-1)+'12']), 0)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'11']-df['preauthorized_payment_cnt'+str(int(year)-1)+'12']), 0)
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'12']-df['cnt_net_transactions'+year+'01']), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'12']-df['sum_net_transactions_amt'+year+'01']), 0)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'12']-df['travel_cnt'+year+'01']), 0)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'12']-df['travel_amt'+year+'01']), 0)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'12']-df['need_cnt'+year+'01']), 0)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'12']-df['sum_net_transactions_amt'+year+'01']), 0)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'12']-df['want_cnt'+year+'01']), 0)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'12']-df['want_amt'+year+'01']), 0)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'12']- df['interchange'+year+'01']), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'12']-df['preauthorized_payment_amt'+year+'01']), 0)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'12']-df['preauthorized_payment_cnt'+year+'01']), 0)

            else:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'11']+df['cnt_net_transactions'+str(int(year)-1)+'12']+df['cnt_net_transactions'+year+'01'])/3, df.match_count_net_transactions_prev_3months)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'11']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['sum_net_transactions_amt'+year+'01'])/3, df.match_sum_net_transactions_amt_prev_3months)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'11']+df['travel_cnt'+str(int(year)-1)+'12']+df['travel_cnt'+year+'01'])/3, df.match_travel_count_prev_3months)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'11']+df['travel_amt'+str(int(year)-1)+'12']+df['travel_amt'+year+'01'])/3, df.match_travel_amount_3months)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'11']+ df['need_cnt'+str(int(year)-1)+'12']+ df['need_cnt'+year+'01'])/3, df.match_need_count_prev_3months)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'11']+df['need_amt'+str(int(year)-1)+'12']+df['need_amt'+year+'01'])/3, df.match_need_amount_prev_3months)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'11']+df['want_cnt'+str(int(year)-1)+'12']+df['want_cnt'+year+'01'])/3, df.match_want_count_prev_3months)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'11']+df['want_amt'+str(int(year)-1)+'12']+df['want_amt'+year+'01'])/3, df.match_want_amount_3months)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'11']+ df['interchange'+str(int(year)-1)+'12']+ df['interchange'+year+'01'])/3, df.match_interchange_prev_3months)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'11']+df['preauthorized_payment_amt'+str(int(year)-1)+'12']+df['preauthorized_payment_amt'+year+'01'])/3, df.match_preauthorized_payment_amt_prev_3months)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'11']+df['preauthorized_payment_cnt'+str(int(year)-1)+'12']+df['preauthorized_payment_cnt'+year+'01'])/3, df.match_preauthorized_payment_cnt_3months)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'11']-df['cnt_net_transactions'+str(int(year)-1)+'12']), df.match_count_net_transactions_prev_3months_delta_3_2)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'11']-df['sum_net_transactions_amt'+str(int(year)-1)+'12']), df.match_sum_net_transactions_amt_prev_3months_delta_3_2)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'11']-df['travel_cnt'+str(int(year)-1)+'12']), df.match_travel_count_prev_3months_delta_3_2)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'11']-df['travel_amt'+str(int(year)-1)+'12']), df.match_travel_amount_3months_delta_3_2)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'11']-df['need_cnt'+str(int(year)-1)+'12']), df.match_need_count_prev_3months_delta_3_2)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'11']-df['sum_net_transactions_amt'+str(int(year)-1)+'12']), df.match_need_amount_prev_3months_delta_3_2)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'11']-df['want_cnt'+str(int(year)-1)+'12']), df.match_want_count_prev_3months_delta_3_2)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'11']-df['want_amt'+str(int(year)-1)+'12']), df.match_want_amount_3months_delta_3_2)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'11']- df['interchange'+str(int(year)-1)+'12']), df.match_interchange_prev_3months_delta_3_2)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'11']-df['preauthorized_payment_amt'+str(int(year)-1)+'12']), df.match_preauthorized_payment_amt_prev_3months_delta_3_2)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'11']-df['preauthorized_payment_cnt'+str(int(year)-1)+'12']), df.match_preauthorized_payment_cnt_3months_delta_3_2)
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(int(year)-1)+'12']-df['cnt_net_transactions'+year+'01']), df.match_count_net_transactions_prev_3months_delta_2_1)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(int(year)-1)+'12']-df['sum_net_transactions_amt'+year+'01']), df.match_sum_net_transactions_amt_prev_3months_delta_2_1)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(int(year)-1)+'12']-df['travel_cnt'+year+'01']), df.match_travel_count_prev_3months_delta_2_1)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(int(year)-1)+'12']-df['travel_amt'+year+'01']), df.match_travel_amount_3months_delta_2_1)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(int(year)-1)+'12']-df['need_cnt'+year+'01']), df.match_need_count_prev_3months_delta_2_1)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(int(year)-1)+'12']-df['sum_net_transactions_amt'+year+'01']), df.match_need_amount_prev_3months_delta_2_1)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(int(year)-1)+'12']-df['want_cnt'+year+'01']), df.match_want_count_prev_3months_delta_2_1)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(int(year)-1)+'12']-df['want_amt'+year+'01']), df.match_want_amount_3months_delta_2_1)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(int(year)-1)+'12']- df['interchange'+year+'01']), df.match_interchange_prev_3months_delta_2_1)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(int(year)-1)+'12']-df['preauthorized_payment_amt'+year+'01']), df.match_preauthorized_payment_amt_prev_3months_delta_2_1)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(int(year)-1)+'12']-df['preauthorized_payment_cnt'+year+'01']), df.match_preauthorized_payment_cnt_3months_delta_2_1)

        elif march:
            if cohort_ind == 0:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+year+'01']+df['cnt_net_transactions'+str(int(year)-1)+'12']+df['cnt_net_transactions'+year+'02'])/3, 0)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+year+'01']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['sum_net_transactions_amt'+year+'02'])/3, 0)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+year+'01']+df['travel_cnt'+str(int(year)-1)+'12']+df['travel_cnt'+year+'02'])/3, 0)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+year+'01']+df['travel_amt'+str(int(year)-1)+'12']+df['travel_amt'+year+'02'])/3, 0)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+year+'01']+df['need_cnt'+str(int(year)-1)+'12']+df['need_cnt'+year+'02'])/3, 0)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+year+'01']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['need_amt'+year+'02'])/3, 0)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+year+'01']+df['want_cnt'+str(int(year)-1)+'12']+df['want_cnt'+year+'02'])/3, 0)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+year+'01']+df['want_amt'+str(int(year)-1)+'12']+df['want_amt'+year+'02'])/3, 0)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+year+'01']+ df['interchange'+str(int(year)-1)+'12']+ df['interchange'+year+'02'])/3, 0)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+year+'01']+df['preauthorized_payment_amt'+str(int(year)-1)+'12']+df['preauthorized_payment_amt'+year+'02'])/3, 0)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+year+'01']+df['preauthorized_payment_cnt'+str(int(year)-1)+'12']+df['preauthorized_payment_cnt'+year+'02'])/3, 0)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['cnt_net_transactions'+year+'01']-df['cnt_net_transactions'+str(int(year)-1)+'12']), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['sum_net_transactions_amt'+year+'01']- df['sum_net_transactions_amt'+str(int(year)-1)+'12']), 0)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['travel_cnt'+year+'01']-df['travel_cnt'+str(int(year)-1)+'12']), 0)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['travel_amt'+year+'01']-df['travel_amt'+str(int(year)-1)+'12']), 0)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['need_cnt'+year+'01']-df['need_cnt'+str(int(year)-1)+'12']), 0)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['need_amt'+year+'01']- df['sum_net_transactions_amt'+str(int(year)-1)+'12']), 0)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['want_cnt'+year+'01']+df['want_cnt'+str(int(year)-1)+'12']), 0)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['want_amt'+year+'01']-df['want_amt'+str(int(year)-1)+'12']), 0)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['interchange'+year+'01']-df['interchange'+str(int(year)-1)+'12']), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['preauthorized_payment_amt'+year+'01']-df['preauthorized_payment_amt'+str(int(year)-1)+'12']), 0)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['preauthorized_payment_cnt'+year+'01']-df['preauthorized_payment_cnt'+str(int(year)-1)+'12']), 0)
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+year+'01']-df['cnt_net_transactions'+year+'02']), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+year+'01']-df['sum_net_transactions_amt'+year+'02']), 0)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+year+'01']-df['travel_cnt'+year+'02']), 0)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+year+'01']-df['travel_amt'+year+'02']), 0)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+year+'01']-df['need_cnt'+year+'02']), 0)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+year+'01']-df['sum_net_transactions_amt'+year+'02']), 0)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+year+'01']-df['want_cnt'+year+'02']), 0)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+year+'01']-df['want_amt'+year+'02']), 0)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+year+'01']-df['interchange'+year+'02']), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+year+'01']-df['preauthorized_payment_amt'+year+'02']), 0)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+year+'01']-df['preauthorized_payment_cnt'+year+'02']), 0)

            else:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+year+'01']+df['cnt_net_transactions'+str(int(year)-1)+'12']+df['cnt_net_transactions'+year+'02'])/3, df.match_count_net_transactions_prev_3months)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+year+'01']+ df['sum_net_transactions_amt'+str(int(year)-1)+'12']+ df['sum_net_transactions_amt'+year+'02'])/3, df.match_sum_net_transactions_amt_prev_3months)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+year+'01']+df['travel_cnt'+str(int(year)-1)+'12']+df['travel_cnt'+year+'02'])/3, df.match_travel_count_prev_3months)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+year+'01']+df['travel_amt'+str(int(year)-1)+'12']+df['travel_amt'+year+'02'])/3, df.match_travel_amount_3months)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+year+'01']+ df['need_cnt'+str(int(year)-1)+'12']+ df['need_cnt'+year+'02'])/3, df.match_need_count_prev_3months)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+year+'01']+df['need_amt'+str(int(year)-1)+'12']+df['need_amt'+year+'02'])/3, df.match_need_amount_prev_3months)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+year+'01']+df['want_cnt'+str(int(year)-1)+'12']+df['want_cnt'+year+'02'])/3, df.match_want_count_prev_3months)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+year+'01']+df['want_amt'+str(int(year)-1)+'12']+df['want_amt'+year+'02'])/3, df.match_want_amount_3months)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+year+'01']+ df['interchange'+str(int(year)-1)+'12']+ df['interchange'+year+'02'])/3, df.match_interchange_prev_3months)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+year+'01']+df['preauthorized_payment_amt'+str(int(year)-1)+'12']+df['preauthorized_payment_amt'+year+'02'])/3, df.match_preauthorized_payment_amt_prev_3months)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+year+'01']+df['preauthorized_payment_cnt'+str(int(year)-1)+'12']+df['preauthorized_payment_cnt'+year+'02'])/3, df.match_preauthorized_payment_cnt_3months)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['cnt_net_transactions'+year+'01']-df['cnt_net_transactions'+str(int(year)-1)+'12']), df.match_count_net_transactions_prev_3months_delta_3_2)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['sum_net_transactions_amt'+year+'01']- df['sum_net_transactions_amt'+str(int(year)-1)+'12']), df.match_sum_net_transactions_amt_prev_3months_delta_3_2)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['travel_cnt'+year+'01']-+df['travel_cnt'+str(int(year)-1)+'12']), df.match_travel_count_prev_3months_delta_3_2)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['travel_amt'+year+'01']-df['travel_amt'+str(int(year)-1)+'12']), df.match_travel_amount_3months_delta_3_2)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['need_cnt'+year+'01']-df['need_cnt'+str(int(year)-1)+'12']), df.match_need_count_prev_3months_delta_3_2)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['need_amt'+year+'01']- df['sum_net_transactions_amt'+str(int(year)-1)+'12']), df.match_need_amount_prev_3months_delta_3_2)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['want_cnt'+year+'01']-df['want_cnt'+str(int(year)-1)+'12']), df.match_want_count_prev_3months_delta_3_2)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['want_amt'+year+'01']-df['want_amt'+str(int(year)-1)+'12']), df.match_want_amount_3months_delta_3_2)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['interchange'+year+'01']- df['interchange'+str(int(year)-1)+'12']), df.match_interchange_prev_3months_delta_3_2)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['preauthorized_payment_amt'+year+'01']-df['preauthorized_payment_amt'+str(int(year)-1)+'12']), df.match_preauthorized_payment_amt_prev_3months_delta_3_2)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], -(df['preauthorized_payment_cnt'+year+'01']-df['preauthorized_payment_cnt'+str(int(year)-1)+'12']), df.match_preauthorized_payment_cnt_3months_delta_3_2)
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+year+'01']-df['cnt_net_transactions'+year+'02']), df.match_count_net_transactions_prev_3months_delta_2_1)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+year+'01']-df['sum_net_transactions_amt'+year+'02']), df.match_sum_net_transactions_amt_prev_3months_delta_2_1)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+year+'01']-df['travel_cnt'+year+'02']), df.match_travel_count_prev_3months_delta_2_1)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+year+'01']-df['travel_amt'+year+'02']), df.match_travel_amount_3months_delta_2_1)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+year+'01']-df['need_cnt'+year+'02']), df.match_need_count_prev_3months_delta_2_1)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+year+'01']-df['sum_net_transactions_amt'+year+'02']), df.match_need_amount_prev_3months_delta_2_1)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+year+'01']-df['want_cnt'+year+'02']), df.match_want_count_prev_3months_delta_2_1)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+year+'01']-df['want_amt'+year+'02']), df.match_want_amount_3months_delta_2_1)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+year+'01']-df['interchange'+year+'02']), df.match_interchange_prev_3months_delta_2_1)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+year+'01']-df['preauthorized_payment_amt'+year+'02']), df.match_preauthorized_payment_amt_prev_3months_delta_2_1)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+year+'01']-df['preauthorized_payment_cnt'+year+'02']), df.match_preauthorized_payment_cnt_3months_delta_2_1)                      
                                                                                                              
        else:
            if cohort_ind == 0:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(cohort_list[cohort_ind]-3)]+df['cnt_net_transactions'+str(cohort_list[cohort_ind]-1)]+df['cnt_net_transactions'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-3)]+ df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-1)]+ df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(cohort_list[cohort_ind]-3)]+df['travel_cnt'+str(cohort_list[cohort_ind]-1)]+df['travel_cnt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(cohort_list[cohort_ind]-3)]+df['travel_amt'+str(cohort_list[cohort_ind]-1)]+df['travel_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(cohort_list[cohort_ind]-3)]+df['need_cnt'+str(cohort_list[cohort_ind]-1)]+df['need_cnt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(cohort_list[cohort_ind]-3)]+ df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-1)]+ df['need_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(cohort_list[cohort_ind]-3)]+df['want_cnt'+str(cohort_list[cohort_ind]-1)]+df['want_cnt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(cohort_list[cohort_ind]-3)]+df['want_amt'+str(cohort_list[cohort_ind]-1)]+df['want_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(cohort_list[cohort_ind]-3)]+ df['interchange'+str(cohort_list[cohort_ind]-1)]+ df['interchange'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-3)]+df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-1)]+df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-3)]+df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-1)]+df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-2)])/3, 0)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(cohort_list[cohort_ind]-3)]-df['cnt_net_transactions'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-3)]- df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(cohort_list[cohort_ind]-3)]-df['travel_cnt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(cohort_list[cohort_ind]-3)]-df['travel_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(cohort_list[cohort_ind]-3)]-df['need_cnt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(cohort_list[cohort_ind]-3)]-df['need_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(cohort_list[cohort_ind]-3)]-df['want_cnt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(cohort_list[cohort_ind]-3)]-df['want_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(cohort_list[cohort_ind]-3)]- df['interchange'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-3)]-df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-2)]), 0)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-3)]-df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-2)]), 0)        
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(cohort_list[cohort_ind]-2)]-df['cnt_net_transactions'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-2)]- df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(cohort_list[cohort_ind]-2)]-df['travel_cnt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(cohort_list[cohort_ind]-2)]-df['travel_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(cohort_list[cohort_ind]-2)]-df['need_cnt'+str(cohort_list[cohort_ind]-1)]), 0)   
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(cohort_list[cohort_ind]-2)]-df['need_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(cohort_list[cohort_ind]-2)]-df['want_cnt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(cohort_list[cohort_ind]-2)]-df['want_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(cohort_list[cohort_ind]-2)]- df['interchange'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-2)]-df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-1)]), 0)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-2)]-df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-1)]), 0)                                                       
            else:
                df['match_count_net_transactions_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(cohort_list[cohort_ind]-3)]+df['cnt_net_transactions'+str(cohort_list[cohort_ind]-1)]+df['cnt_net_transactions'+str(cohort_list[cohort_ind]-2)])/3, df.match_count_net_transactions_prev_3months)
                df['match_sum_net_transactions_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-3)]+ df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-1)]+ df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_sum_net_transactions_amt_prev_3months)
                df['match_travel_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(cohort_list[cohort_ind]-3)]+df['travel_cnt'+str(cohort_list[cohort_ind]-1)]+df['travel_cnt'+str(cohort_list[cohort_ind]-2)])/3, df.match_travel_count_prev_3months)
                df['match_travel_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(cohort_list[cohort_ind]-3)]+df['travel_amt'+str(cohort_list[cohort_ind]-1)]+df['travel_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_travel_amount_3months)
                df['match_need_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(cohort_list[cohort_ind]-3)]+ df['need_cnt'+str(cohort_list[cohort_ind]-1)]+ df['need_cnt'+str(cohort_list[cohort_ind]-2)])/3, df.match_need_count_prev_3months)
                df['match_need_amount_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(cohort_list[cohort_ind]-3)]+df['need_amt'+str(cohort_list[cohort_ind]-1)]+df['need_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_need_amount_prev_3months)
                df['match_want_count_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(cohort_list[cohort_ind]-3)]+df['want_cnt'+str(cohort_list[cohort_ind]-1)]+df['want_cnt'+str(cohort_list[cohort_ind]-2)])/3, df.match_want_count_prev_3months)
                df['match_want_amount_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(cohort_list[cohort_ind]-3)]+df['want_amt'+str(cohort_list[cohort_ind]-1)]+df['want_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_want_amount_3months)
                df['match_interchange_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(cohort_list[cohort_ind]-3)]+ df['interchange'+str(cohort_list[cohort_ind]-1)]+ df['interchange'+str(cohort_list[cohort_ind]-2)])/3, df.match_interchange_prev_3months)
                df['match_preauthorized_payment_amt_prev_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-3)]+df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-1)]+df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-2)])/3, df.match_preauthorized_payment_amt_prev_3months)
                df['match_preauthorized_payment_cnt_3months'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-3)]+df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-1)]+df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-2)])/3, df.match_preauthorized_payment_cnt_3months)
                df['match_count_net_transactions_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(cohort_list[cohort_ind]-3)]-df['cnt_net_transactions'+str(cohort_list[cohort_ind]-2)]), df.match_count_net_transactions_prev_3months_delta_3_2)
                df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-3)]- df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-2)]), df.match_sum_net_transactions_amt_prev_3months_delta_3_2)
                df['match_travel_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(cohort_list[cohort_ind]-3)]-df['travel_cnt'+str(cohort_list[cohort_ind]-2)]), df.match_travel_count_prev_3months_delta_3_2)
                df['match_travel_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(cohort_list[cohort_ind]-3)]-df['travel_amt'+str(cohort_list[cohort_ind]-2)]), df.match_travel_amount_3months_delta_3_2)
                df['match_need_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(cohort_list[cohort_ind]-3)]-df['need_cnt'+str(cohort_list[cohort_ind]-2)]), df.match_need_count_prev_3months_delta_3_2)
                df['match_need_amount_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(cohort_list[cohort_ind]-3)]-df['need_amt'+str(cohort_list[cohort_ind]-2)]), df.match_need_amount_prev_3months_delta_3_2)
                df['match_want_count_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(cohort_list[cohort_ind]-3)]-df['want_cnt'+str(cohort_list[cohort_ind]-2)]), df.match_want_count_prev_3months_delta_3_2)
                df['match_want_amount_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(cohort_list[cohort_ind]-3)]-df['want_amt'+str(cohort_list[cohort_ind]-2)]), df.match_want_amount_3months_delta_3_2)
                df['match_interchange_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(cohort_list[cohort_ind]-3)]- df['interchange'+str(cohort_list[cohort_ind]-2)]), df.match_interchange_prev_3months_delta_3_2)
                df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-3)]-df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-2)]), df.match_preauthorized_payment_amt_prev_3months_delta_3_2)
                df['match_preauthorized_payment_cnt_3months_delta_3_2'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-3)]-df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-2)]), df.match_preauthorized_payment_cnt_3months_delta_3_2)                                                                                                    
                df['match_count_net_transactions_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['cnt_net_transactions'+str(cohort_list[cohort_ind]-2)]-df['cnt_net_transactions'+str(cohort_list[cohort_ind]-1)]), df.match_count_net_transactions_prev_3months_delta_2_1)
                df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-2)]- df['sum_net_transactions_amt'+str(cohort_list[cohort_ind]-1)]), df.match_sum_net_transactions_amt_prev_3months_delta_2_1)
                df['match_travel_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_cnt'+str(cohort_list[cohort_ind]-2)]-df['travel_cnt'+str(cohort_list[cohort_ind]-1)]), df.match_travel_count_prev_3months_delta_2_1)
                df['match_travel_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['travel_amt'+str(cohort_list[cohort_ind]-2)]-df['travel_amt'+str(cohort_list[cohort_ind]-1)]), df.match_travel_amount_3months_delta_2_1)
                df['match_need_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_cnt'+str(cohort_list[cohort_ind]-2)]-df['need_cnt'+str(cohort_list[cohort_ind]-1)]), df.match_need_count_prev_3months_delta_2_1)
                df['match_need_amount_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['need_amt'+str(cohort_list[cohort_ind]-2)]-df['need_amt'+str(cohort_list[cohort_ind]-1)]), df.match_need_amount_prev_3months_delta_2_1)
                df['match_want_count_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_cnt'+str(cohort_list[cohort_ind]-2)]-df['want_cnt'+str(cohort_list[cohort_ind]-1)]), df.match_want_count_prev_3months_delta_2_1)
                df['match_want_amount_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['want_amt'+str(cohort_list[cohort_ind]-2)]-df['want_amt'+str(cohort_list[cohort_ind]-1)]), df.match_want_amount_3months_delta_2_1)
                df['match_interchange_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['interchange'+str(cohort_list[cohort_ind]-2)]- df['interchange'+str(cohort_list[cohort_ind]-1)]), df.match_interchange_prev_3months_delta_2_1)
                df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-2)]-df['preauthorized_payment_amt'+str(cohort_list[cohort_ind]-1)]), df.match_preauthorized_payment_amt_prev_3months_delta_2_1)
                df['match_preauthorized_payment_cnt_3months_delta_2_1'] = np.where(df.cohort == cohort_list[cohort_ind], (df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-2)]-df['preauthorized_payment_cnt'+str(cohort_list[cohort_ind]-1)]), df.match_preauthorized_payment_cnt_3months_delta_2_1)
                                                                                                    
                                                                                                    
    #set all the new columns to int32 instead of int64
    df['match_count_net_transactions_prev_3months'] = df['match_count_net_transactions_prev_3months'].astype('int32')
    df['match_sum_net_transactions_amt_prev_3months'] = df['match_sum_net_transactions_amt_prev_3months'].astype('int32')
    df['match_travel_count_prev_3months'] = df['match_travel_count_prev_3months'].astype('int32')
    df['match_travel_amount_3months'] = df['match_travel_amount_3months'].astype('int32')
    df['match_need_count_prev_3months'] = df['match_need_count_prev_3months'].astype('int32')
    df['match_need_amount_prev_3months'] = df['match_need_amount_prev_3months'].astype('int32')
    df['match_want_count_prev_3months'] = df['match_want_count_prev_3months'].astype('int32')
    df['match_want_amount_3months'] = df['match_want_amount_3months'].astype('int32')
    df['match_interchange_prev_3months'] = df['match_interchange_prev_3months'].astype('int32')
    df['match_preauthorized_payment_amt_prev_3months'] = df['match_preauthorized_payment_amt_prev_3months'].astype('int32')
    df['match_preauthorized_payment_cnt_3months'] = df['match_preauthorized_payment_cnt_3months'].astype('int32')
    df['match_count_net_transactions_prev_3months_delta_3_2'] = df['match_count_net_transactions_prev_3months_delta_3_2'].astype('float32')
    df['match_sum_net_transactions_amt_prev_3months_delta_3_2'] = df['match_sum_net_transactions_amt_prev_3months_delta_3_2'].astype('float32')
    df['match_travel_count_prev_3months_delta_3_2'] = df['match_travel_count_prev_3months_delta_3_2'].astype('float32')
    df['match_travel_amount_3months_delta_3_2'] = df['match_travel_amount_3months_delta_3_2'].astype('float32')
    df['match_need_count_prev_3months_delta_3_2'] = df['match_need_count_prev_3months_delta_3_2'].astype('float32')
    df['match_need_amount_prev_3months_delta_3_2'] = df['match_need_amount_prev_3months_delta_3_2'].astype('float32')
    df['match_want_count_prev_3months_delta_3_2'] = df['match_want_count_prev_3months_delta_3_2'].astype('float32')
    df['match_want_amount_3months_delta_3_2'] = df['match_want_amount_3months_delta_3_2'].astype('float32')
    df['match_interchange_prev_3months_delta_3_2'] = df['match_interchange_prev_3months_delta_3_2'].astype('float32')
    df['match_preauthorized_payment_amt_prev_3months_delta_3_2'] = df['match_preauthorized_payment_amt_prev_3months_delta_3_2'].astype('float32')
    df['match_preauthorized_payment_cnt_3months_delta_3_2'] = df['match_preauthorized_payment_cnt_3months_delta_3_2'].astype('float32')
    df['match_count_net_transactions_prev_3months_delta_2_1'] = df['match_count_net_transactions_prev_3months_delta_2_1'].astype('float32')
    df['match_sum_net_transactions_amt_prev_3months_delta_2_1'] = df['match_sum_net_transactions_amt_prev_3months_delta_2_1'].astype('float32')
    df['match_travel_count_prev_3months_delta_2_1'] = df['match_travel_count_prev_3months_delta_2_1'].astype('float32')
    df['match_travel_amount_3months_delta_2_1'] = df['match_travel_amount_3months_delta_2_1'].astype('float32')
    df['match_need_count_prev_3months_delta_2_1'] = df['match_need_count_prev_3months_delta_2_1'].astype('float32')
    df['match_need_amount_prev_3months_delta_2_1'] = df['match_need_amount_prev_3months_delta_2_1'].astype('float32')
    df['match_want_count_prev_3months_delta_2_1'] = df['match_want_count_prev_3months_delta_2_1'].astype('float32')
    df['match_want_amount_3months_delta_2_1'] = df['match_want_amount_3months_delta_2_1'].astype('float32')
    df['match_interchange_prev_3months_delta_2_1'] = df['match_interchange_prev_3months_delta_2_1'].astype('float32')
    df['match_preauthorized_payment_amt_prev_3months_delta_2_1'] = df['match_preauthorized_payment_amt_prev_3months_delta_2_1'].astype('float32')
    df['match_preauthorized_payment_cnt_3months_delta_2_1'] = df['match_preauthorized_payment_cnt_3months_delta_2_1'].astype('float32')                                                                                 

    #Remove index columns not required anymore
    df = df.drop(cnt_net_trans_list+net_trans_amt_list+trav_count_list+trav_amt_list+need_cnt_list+need_amt_list+want_count_list+want_amt_list+interchange_list + preauth_amt_count_list+preauth_cnt_list, axis = 1)
    
    return df


def transform(df, emerald, aero, prod):
    '''
    Main Function which does all the transformations to the input dataframe
    input: dataframe
    output: transformed dataframe
    effect: mutates input dataframe
    '''
    #define list of cpcs
    print('Running in Production: ', prod)
    cpc_list = [col for col in df.columns if 'cpc2' in col]
    print(cpc_list)

    from datetime import datetime

    cpc_list_dates = [datetime.strptime(x, '%Y%m') for x in list(map(lambda x: x.strip('cpc'),cpc_list))]
    most_recent_date = cpc_list_dates[-1]
    #Cut off date is one year prior to most recent date
    cutoff_date = datetime.strptime((str(most_recent_date.year-1)+str(most_recent_date.month)), '%Y%m')

    #filter cpc list from cutoff date onwards
    new_cpc_list = []
    for i in range(len(cpc_list_dates)):
        if cpc_list_dates[i]>=cutoff_date:
            new_cpc_list.append(cpc_list[i])


    df = df.fillna(0)

    #Remove Pure Gamers (>2 unique products in the past year)

    if not prod:
        print('Shape of dataframe priot to gamer removal:  ', df.shape)

    df = remove_pure_gamers(df, new_cpc_list)

    if not prod:
        print('Pure Gamers Removed')
        print('Shape After Pure Gamer removal: ', df.shape)

    #Add flag for no migration
    df = no_migration_flag(df, new_cpc_list)

    if not prod:
        print('No Migration Flag added')

    #Add flag for soft gamers
    df = tag_softgamers(df, new_cpc_list)

    #Remove soft gamers and the tag
    df = df[df.soft_gamers==0]

    df = df.drop('soft_gamers', axis=1)

    if not prod:
        print('Soft Gamers Removed')
        print('Shape after Soft Gamers Removal', df.shape)

    #define cohort 
    df_nogame = cohort_df(df, new_cpc_list)

    if not prod:
        print('cohort defined')

    if not prod:
        print (new_cpc_list)

    if not prod:
        print (df_nogame['cohort'].head(3))

    df_nogame['cohort']=np.where(df_nogame['cohort'] == 0, np.uint32(201908), df.cohort)
    df_nogame['cohort_year'] = pd.to_datetime(df_nogame.cohort, format = '%Y%m').dt.year.astype('int32')
    df_nogame['cohort_month']  = pd.to_datetime(df_nogame.cohort, format = '%Y%m').dt.month.astype('int32')

    if not prod:
        print (df_nogame[['cohort','cohort_year', 'cohort_month']].head(3))
        print(df_nogame.shape)
        
    #Remove Accts: 19('Bad'), 20('Charge Off') and 22('Fraud') 
    #Keep Accts: 23('Dormant') and 24('Active')
    A_Status_df = df_nogame[df_nogame.acct_status.isin(['23','24'])]

    df_acstat = pd.concat([A_Status_df, pd.get_dummies(A_Status_df.acct_status, dtype = 'int32')], axis=1)

    #Change columns names from 23, 24 to active or dormant
    df_acstat = df_acstat.rename(columns={23: 'match_acct_status_dormant', 24: 'match_acct_status_active'})

    df_astatdummy = df_acstat.drop('acct_status',axis=1)

    #Add Migration Type
    df_mig = Migration_type(df_astatdummy, new_cpc_list)

    #Add cpc_before and After
    df_cpc = cpc_before_after(df_mig, new_cpc_list)

    if not prod:
        print('cpc before and After defined')

    if not prod:
        print (df_cpc[['cpc_before', 'match_cpc_after']].head(3))
        print(df_cpc.shape)

    #Add tag to indetify production dataset to predict for
    df_cpc = production_flag_cohort(df_cpc)


    if not prod:
        print('Unique vals Prod Tag:  ',set(df_cpc['match_prod_tag'].unique()))
        print('Shape of df prior no mig removal:  ',df_cpc.shape)

    #Uncomment below to remove No Migrations from the whole dataset below. 
    df_cpc = remove_no_migration(df_cpc)

    if not prod:
        print('Unique vals Prod Tag After Removal:  ',set(df_cpc['match_prod_tag'].unique()))
        print('Shape of df After no mig removal:  ',df_cpc.shape)

    if not prod:
        print('prod flag added, no migrations removed')

    if not prod:
        print (df_cpc[['cohort','cohort_month','cohort_year','match_prod_tag','cpc_before', 'match_cpc_after']].head(3))
        print(df_cpc.shape)

    #Remove Aeroplan cpc
    if aero:
        print('Keeping Aeroplan customers')
    else:
        df_cpc = remove_aeroplan_after(df_cpc)
        print('AeroPlan cards removed')

    if emerald:
        print('Keeping Emerald customers')
    else:
        df_cpc = remove_emerald(df_cpc)
        print('Emerald_cards removed')

    df_cpc = remove_old_cards(df_cpc)

    if not prod:
        print('Old Cards Removed')

    df_days = dummy_cpc_before(df_cpc)

    if not prod:
        print('Dummy cpc Before created')

    #Since FIW and SOW data is bimonthly, using next months data to create previous months data for missing months (Backward Fill)
    df_days['fiw_flg_201807'] = df_days['fiw_flg_201808']
    df_days['fiw_flg_201809'] = df_days['fiw_flg_201810']
    df_days['fiw_flg_201811'] = df_days['fiw_flg_201812']
    df_days['fiw_flg_201901'] = df_days['fiw_flg_201902']
    df_days['fiw_flg_201903'] = df_days['fiw_flg_201904']
    df_days['fiw_flg_201905'] = df_days['fiw_flg_201906']
    df_days['fiw_flg_201907'] = df_days['fiw_flg_201908']

    df_days['sow_cl_201807'] = df_days['sow_cl_201808']
    df_days['sow_cl_201809'] = df_days['sow_cl_201810']
    df_days['sow_cl_201811'] = df_days['sow_cl_201812']
    df_days['sow_cl_201901'] = df_days['sow_cl_201902']
    df_days['sow_cl_201903'] = df_days['sow_cl_201904']
    df_days['sow_cl_201905'] = df_days['sow_cl_201906']
    df_days['sow_cl_201907'] = df_days['sow_cl_201908']

    df_days['sow_bal_201807'] = df_days['sow_bal_201808']
    df_days['sow_bal_201809'] = df_days['sow_bal_201810']
    df_days['sow_bal_201811'] = df_days['sow_bal_201812']
    df_days['sow_bal_201901'] = df_days['sow_bal_201902']
    df_days['sow_bal_201903'] = df_days['sow_bal_201904']
    df_days['sow_bal_201905'] = df_days['sow_bal_201906']
    df_days['sow_bal_201907'] = df_days['sow_bal_201908']

    #Add credit_score etc 
    df_days = match_Ind(df_days)

    if not prod:
        print('match Ind done')
        print(df_days.head(3))

    #Add if customer had other kinds of accounts prior to migration
    df_acct_inds = acct_index_pre_move(df_days)

    #Annual Fee, AutoPay, Change Date, Cred Limit Change added
    #uncomment the code below if the fields needed for match_Amt has data for all months (201708 onwards)
    df_fees = match_Amt(df_acct_inds)

    #Add days passed since open date feature
    df_acct_inds = opendate_to_days(df_fees)

    if not prod:
        print('Days since open date DONE')

    if not prod:
        print('Acct Index Pre Move and match_amt done')
        print (df_acct_inds.head(3))
        print(df_acct_inds.shape)

    #Transaction features added
    df_trans = transaction_acct(df_acct_inds)

    #Rewards, aeroplan rewards and apple pay added
    df_rewards = rewards_acct(df_trans)

    if not prod:
        print('Transactions and rewards done')
        print (df_rewards.filter(regex='match').head(3))
        print(df_rewards.shape)

    #Add Cash features
    df_cash = acct_cash_feats(df_rewards)

    #Add financial features (eg. cred limit, adb, etc)
    df_cash = avg_three_months_prior_financial(df_cash)

    #Add basic customer indexes (eg. student index, easyweb index. currently province index is 
    # commented out because I'm not sure how to encode it due to high cardinality)
    df_cash = index_customer_basics(df_cash)

    #Add feature for campaigns running and join with datekey
    
    #

    if not prod:
        print('Cash features added')

    df_cash = df_cash.drop(['cohort'], axis=1)

    if not prod:
        print('Cohort dropped')

    match_columns = [column for column in df_cash.columns if 'match' in column]

    df_final = pd.concat([df_cash[match_columns], df_cash[['cohort_year', 'cohort_month', 'tsys_acct_id']]], axis= 1)

    #Remove MMC and TAW as cpc_before
    #df_final = df_final[(df_final['match_cpc_before_TAW']!=1)&(df_final['match_cpc_before_MMC']!=1)]
    #df_final = df_final.drop(['match_cpc_before_MMC','match_cpc_before_TAW'],axis=1)
                             
    if not prod:
        print('Final Dataframe is:/n')
        print (df_final.head(3))
        print(df_final.shape)

    return df_final


if __name__ == "__main__":

    #Set this to output the whole dataframe without truncating
    pd.set_option('display.max_columns', None)

    parser = argparse.ArgumentParser(description='Flags for which data to remove and keep')

    parser.add_argument('-keep-emerald', action='store_true', default=False,
                        dest='emerald',
                        help='Set to keep emerald')

    parser.add_argument('-keep-aero', action='store_true', default=False,
                        dest='aero',
                        help='Set to keep Aeroplan')
    
    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)

    parser.add_argument('--prod', action='store_true', default=False,
                        dest='prod',
                        help='Set to run in production. ie, will not print debug statements')

    results = parser.parse_args()

    if not (os.path.isfile('../Data/ORD.csv')):
        print ("Data file does NOT exist")
        sys.exit(10)

    import time

    start = time.time()
                                            
    # read the large csv file with specified chunksize
    df_chunk = pd.read_csv('../Data/ORD.csv', chunksize=100000)

    chunk_list = []  # append each chunk df here 

    # Each chunk is in df format
    for chunk in df_chunk:  
        # Once the data filtering is done, append the chunk to list
        chunk_list.append(chunk)
        
    # concat the list into dataframe 
    df = pd.concat(chunk_list)

    print("Reading in dataframe took %.2f seconds" % ((time.time() - start)))

    start2= time.time()

    n = 10000  #chunk row size
    list_df = [df[i:i+n] for i in range(0,df.shape[0],n)]

    print("Partitioning dataframe took %.2f seconds" % ((time.time() - start2)))

    #Transforming data in chunks to not have Memory Error
    list_df_trans = []
    print('Number of chunks:',len(list_df))
    for i in range(len(list_df)):
        print('Starting Chunk number:', i)
        list_df[i].columns = list_df[i].columns.str.lower()
        df_transformed = transform(list_df[i], results.emerald, results.aero, results.prod)
        list_df_trans.append(df_transformed)

    df_transformed_final = pd.concat(list_df_trans,axis=0, ignore_index=True)

    df_transformed_final = df_transformed_final[(df_transformed_final['match_cpc_before_TAW']!=1)]
    df_transformed_final = df_transformed_final.drop('match_cpc_before_TAW',axis=1)
    #Remove TGC from CPC After
    df_transformed_final = df_transformed_final[df_transformed_final['match_cpc_after']!='TGC']

    df_transformed_final['match_cpc_after'].fillna('', inplace=True)

    print('# of rows pre NA fill',df_transformed_final.shape[0])
    print('# of rows without NA',df_transformed_final.dropna().shape[0])
    print('# of rows with NULL values',df_transformed_final.shape[0]-df_transformed_final.dropna().shape[0])

    df_transformed_final = df_transformed_final.fillna(0)

    #Store dataframe as hdf5 file
    df_transformed_final.to_hdf('../'+results.date+'/Data/df_transform.h5', 'df', format='t', complevel=5, complib='bzip2')
