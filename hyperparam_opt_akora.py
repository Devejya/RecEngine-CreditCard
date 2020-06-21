import pandas as pd 
import numpy as np 
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from sklearn import preprocessing
import json
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import os.path
import sys
import argparse
from pyspark.context import SparkContext
from pyspark.sql import HiveContext, SparkSession


def objective(space):
    '''
    Obejctive Function for HyperParameter Optimization for XGB MultiClass Classification Problem. Returns the best score found
    '''
    global best_score
    
    model = XGBClassifier(objective='multi:softprob', num_class=8, seed=0, n_jobs=-1)
    model.set_params(**space)

    kfold = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='neg_log_loss', verbose=True).mean()
    
    scores.append(score)
    #max since neg_logloss returns negative of logloss and smaller logloss is better
    best_score = max(scores)
    
    #return negative of best_score, since we want to minimise the positive logloss tending to 0
    return -best_score

class NpEncoder(json.JSONEncoder):
    '''
    Encoder to Convert numpy types to python dtypes for JSON
    '''
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    #Set this to output the whole dataframe without truncating

    parser = argparse.ArgumentParser(description='Get date to access data file')

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)
    
    parser.add_argument('--prod', action='store_true', default=False,
                        dest='prod',
                        help='Set to run in production. ie, will not print debug statements')

    results = parser.parse_args()

    
#     if not (os.path.isfile(results.date+'/Data/df_correlation_removed.h5')):
#         print ("Data Correlation Removed file does NOT exist")
#         sys.exit(5)
    
    
    #df = pd.read_hdf( results.date+'/Data/df_correlation_removed.h5', 'df')
    #df = pd.read_hdf( results.date+'/Data/df_transformed_fixed.h5', 'df')
    #df = pd.read_hdf( results.date+'/Data/df_tpb_no_tgc.h5', 'df')
    #df = pd.read_hdf( results.date+'/Data/df_transformed_prod.h5', 'df')
    sc = SparkContext(appName="Prem_Data_Transform_")
    # Optional creation of a HiveContext
    sql_context = HiveContext(sc)
    # Optional creation of a SparkSession
    spark = SparkSession(sc)
    spark = (SparkSession.builder.enableHiveSupport().getOrCreate())  
    
    spark_df = spark.read.table("anp_camktedw1_sandbox.jai_prem_transformed")
    # Enable Arrow-based columnar data transfers
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    # Convert the Spark DataFrame to a pandas DataFrame using Arrow
    df = spark_df.select("*").toPandas()
    
    #Some Cleaning for temporary measure
    df = df.drop_duplicates()
    df = df[df['match_cpc_after']!='TGC']

    print(df.head())

    print('Main DataFrame:')
    print(df.head(3))
    print(df.shape)

    #Take out Production Data Set and write to file
    df_prod = df[df['match_prod_tag']==1].drop('match_prod_tag', axis=1)
    df_prod.to_hdf( results.date+'/Data/df_production.h5', 'df', format='t', complevel=5, complib='bzip2')

    print('Production Data:')
    print(df_prod.head())
    print(df.shape)

    #Stratified Split train and test and store test data in file
    x = df[df['match_prod_tag']==0].drop(['match_prod_tag', 'match_cpc_after', 'tsys_acct_id'], axis=1)
    #x = df[df['match_prod_tag']==0].drop(['match_prod_tag', 'match_cpc_after'], axis=1)
    y = df[df['match_prod_tag']==0].drop('match_prod_tag', axis=1)['match_cpc_after']

    print('X is: ', x.head())
    print('Y is: ',y.head())

    #Label Encode the Target (y)
    #le = preprocessing.LabelEncoder()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify = y)

    if not results.prod:
        print('x_train is: ', x_train.head())
        print('y_train is: ', y_train.head())
        print('x_test is: ', x_test.head())
        print('y_test is: ', y_test.head())
        print('x_train shape is: ', x_train.shape)
        print('y_train shape is: ', y_train.shape)
        print('x_test shape is: ', x_test.shape)
        print('y_test shape is: ', y_test.shape)
        print('Unique values in y_train: ', y_train.unique())
        print('Unique values in y_test: ', y_test.unique())

    test_set = pd.concat([x_test, pd.Series(y_test, name='match_cpc_after')], axis=1)

    #Store Test Set as hdf5 file
    test_set.to_hdf( results.date+'/Data/Test_Set.h5', 'df', format='t', complevel=5, complib='bzip2')

    train_set = pd.concat([x_train, pd.Series(y_train, name='match_cpc_after')], axis=1)

    #Store Train Set as hdf5 file
    train_set.to_hdf( results.date+'/Data/Train_Set.h5', 'df', format='t', complevel=5, complib='bzip2')


    # Declare xgboost search space for Hyperopt
    
    #'eta':hp.choice('x_eta',np.arange(0,1,0.01)),
    #'gamma':hp.choice('x_gamma',np.arange(0,200,10)),
    #'alpha':hp.choice('x_alpha',np.arange(0,1,0.01)),
    #'lambda':hp.choice('x_lambda', np.arange(0,1,0.01)),
                    
    xgboost_space={
               'eval_metric': hp.choice('x_eval_metric', ['merror','mlogloss']),
                'max_depth': hp.choice('x_max_depth',[3,4,5,6,7,8,10]),
                'min_child_weight':hp.choice('x_min_child_weight',np.round(np.arange(0.0,0.2,0.01),5)),
                'learning_rate':hp.choice('x_learning_rate',np.round(np.arange(0.005,0.3,0.01),5)),
               'subsample':hp.choice('x_subsample',np.round(np.arange(0.1,1.0,0.05),5)),
                'colsample_bylevel':hp.choice('x_colsample_bylevel',np.round(np.arange(0.1,1.0,0.05),5)),
                'colsample_bytree':hp.choice('x_colsample_bytree',np.round(np.arange(0.1,1.0,0.05),5)),
                'n_estimators':hp.choice('x_n_estimators',np.arange(100,300,40)),
                }

    scores = []
    start = time.time()

    best = fmin(
    objective, 
    space = xgboost_space, 
    algo = tpe.suggest, 
    max_evals = 100,
    trials = Trials())

    best_params = space_eval(xgboost_space, best)

    print("Hyperopt search took %.2f seconds" % ((time.time() - start)))
    print("Best score: %.2f " % (-best_score))
    print("Best space: ", best_params)

    best_score_dict = {'LogLoss': float(best_score)}

    print(best_params)
    #Store the best parameters found as JSON files since JSON files are basically in forms of dictionaries
    with open( results.date+'/Data/Best_Params.json', 'w') as fp:
        json.dump(best_params, fp, sort_keys=True, indent=4, cls=NpEncoder)

    print(best_score_dict)
    #Store the best score from 5Fold CV given the best Parameters as a JSON file since best_score is stored as a dictionary
    with open( results.date+'/Data/Best_Score.json', 'w') as fp:
        json.dump(best_score_dict, fp, sort_keys=True, indent=4, cls=NpEncoder)
