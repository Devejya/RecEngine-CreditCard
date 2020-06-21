'''
Correlation Analysis
'''
#import packages
import pandas as pd 
import numpy as np 
import itertools
import os.path
import sys
import argparse


def identify_correlated_features(X):
    '''
    Performs Pearson Correlation on input dataframe and returns a dataframe with the pairs and there correlation coefficient
    Requires: variables in X should be numeric or binary
    Input: Dataframe
    Output: Dataframe
    '''

    df = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])

    return df.sort_values(by='corr',ascending=False)


def remove_correlated_features(df, coeff):
    '''
    returns a dataframe with values as the names of features to remove that are correlated higher than the coeff given.
    '''

    df_correlated_pairs = pd.DataFrame(df[df['corr']>coeff]['pairs'].values.tolist(), columns = ['Keep', 'Remove'])

    df_keep_features = df_correlated_pairs['Keep']
    df_remove_features = df_correlated_pairs['Remove']

    df_actual_remove = pd.Series(np.where(~df_remove_features.isin(df_keep_features), df_remove_features, None), name='Remove')
    df_actual_remove = df_actual_remove.dropna()

    print('Pairs of Features with higher than threshold Correlation Coefficient')
    print(df_correlated_pairs)

    print('Keeping the Following features:\n')
    print(df_keep_features)

    print('\n Removing the following features\n')
    print(df_actual_remove)

    return df_remove_features


if __name__ == "__main__":
    #Set this to output the whole dataframe without truncating
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    parser = argparse.ArgumentParser(description='Required date and coefficient threshold')

    parser.add_argument('--coeff', dest = 'coeff',  action='store', type=float, default = 0.85)

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)

    parser.add_argument('--prod', action='store_true', default=False,
                        dest='prod',
                        help='Set to run in production. ie, will not print debug statements')

    results = parser.parse_args()

    if not (os.path.isfile('../'+results.date+'/Data/df_transform.h5')):
        print ("Data Transform file does NOT exist")
        sys.exit(20)

    df = pd.read_hdf('../'+results.date+'/Data/df_transform.h5', 'df')

    #uncomment below when df_transform has tsys_acct_id
    df_1 = identify_correlated_features(df.drop(['cohort_year', 'cohort_month', 'Match_CPC_After', 'Match_Prod_Tag', 'Tsys_Acct_ID'],axis=1))

    df_final = df.drop(remove_correlated_features(df_1, results.coeff).to_list(), axis=1)

    #Store dataframe as hdf5 file
    df_final.to_hdf('../'+results.date+'/Data/df_correlation_removed.h5', 'df', format='t', complevel=5, complib='bzip2')
