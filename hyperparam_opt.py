import pandas as pd 
import numpy as np 
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
import os.path
import sys
import argparse


def objective(space):
    '''
    Obejctive Function for HyperParameter Optimization for XGB MultiClass Classification Problem. Returns the best ROC-AUC score found
    '''
    global best_score
    
    model = XGBClassifier(objective='multi:softmax', num_class=9, seed=0, eval_metric='mlogloss', n_jobs=-1)
    model.set_params(**space)

    kfold = KFold(n_splits=3, random_state=42, shuffle=True)
    score = cross_val_score(model, x_train, y_train, cv=kfold, scoring='f1_weighted', verbose=False).mean()
    
    scores.append(score)

    best_score = max(scores)
    
    return -best_score


if __name__ == "__main__":
    #Set this to output the whole dataframe without truncating

    parser = argparse.ArgumentParser(description='Get date to access data file')

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)
    
    parser.add_argument('--prod', action='store_true', default=False,
                        dest='prod',
                        help='Set to run in production. ie, will not print debug statements')

    results = parser.parse_args()

    if not (os.path.isfile('../'+results.date+'/Data/df_correlation_removed.h5')):
        print ("Data Correlation Removed file does NOT exist")
        sys.exit(5)

    df = pd.read_hdf('../'+results.date+'/Data/df_correlation_removed.h5', 'df')

    print('Main DataFrame:')
    print(df.head(3))
    print(df.shape)

    #Take out Production Data Set and write to file
    df_prod = df[df['Match_Prod_Tag']==1].drop('Match_Prod_Tag', axis=1)
    df_prod.to_hdf('../'+results.date+'/Data/df_production.h5', 'df', format='t', complevel=5, complib='bzip2')

    print('Production Data:')
    print(df_prod.head())
    print(df.shape)

    #Stratified Split train and test and store test data in file
    x = df[df['Match_Prod_Tag']==0].drop(['Match_Prod_Tag', 'Match_CPC_After', 'Tsys_Acct_ID'], axis=1)
    #x = df[df['Match_Prod_Tag']==0].drop(['Match_Prod_Tag', 'Match_CPC_After'], axis=1)
    y = df[df['Match_Prod_Tag']==0].drop('Match_Prod_Tag', axis=1)['Match_CPC_After']

    print('X is: ', x.head())
    print('Y is: ',y.head())

    #Label Encode the Target (y)
    le = preprocessing.LabelEncoder()

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


    y_train = le.fit_transform(y_train)

    if not results.prod:
        print('Transformed y_train: ', pd.Series(y_train).head())

    y_test = le.transform(y_test)

    if not results.prod:
        print('Transformed y_test: ', pd.Series(y_test).head())

    #exporting the target encoder
    output = open('../'+results.date+'/Data/Target_encoder.p', 'wb')
    pickle.dump(le, output)
    output.close()

    test_set = pd.concat([x_test, pd.Series(y_test, name='Match_CPC_After')], axis=1)

    #Store Test Set as hdf5 file
    test_set.to_hdf('../'+results.date+'/Data/Test_Set.h5', 'df', format='t', complevel=5, complib='bzip2')

    train_set = pd.concat([x_train, pd.Series(y_train, name='Match_CPC_After')], axis=1)

    #Store Train Set as hdf5 file
    train_set.to_hdf('../'+results.date+'/Data/Train_Set.h5', 'df', format='t', complevel=5, complib='bzip2')


    # Declare xgboost search space for Hyperopt
    xgboost_space={
                'max_depth': hp.choice('x_max_depth',[2,3,4,5,6]),
                'min_child_weight':hp.choice('x_min_child_weight',np.round(np.arange(0.0,0.2,0.01),5)),
                'learning_rate':hp.choice('x_learning_rate',np.round(np.arange(0.005,0.3,0.01),5)),
                'subsample':hp.choice('x_subsample',np.round(np.arange(0.1,1.0,0.05),5)),
                'colsample_bylevel':hp.choice('x_colsample_bylevel',np.round(np.arange(0.1,1.0,0.05),5)),
                'colsample_bytree':hp.choice('x_colsample_bytree',np.round(np.arange(0.1,1.0,0.05),5)),
                'n_estimators':hp.choice('x_n_estimators',np.arange(25,100,5)),
                }

    scores = []
    start = time.time()

    best = fmin(
    objective, 
    space = xgboost_space, 
    algo = tpe.suggest, 
    max_evals = 200,
    trials = Trials())

    best_params = space_eval(xgboost_space, best)

    print("Hyperopt search took %.2f seconds" % ((time.time() - start)))
    print("Best score: %.2f " % (-best_score))
    Print("Best space: ", best_params)

    best_score_dict = {'AUC-Mean': best_score}

    #Store the best parameters found
    with open('../'+results.date+'/Data/Best_Params.json', 'w') as fp:
        json.dump(best_params, fp, sort_keys=True, indent=4)

    #Store the best score from 5Fold CV given the best Parameters
    with open('../'+results.date+'/Data/Best_Score.json', 'w') as fp:
        json.dump(best_score_dict, fp, sort_keys=True, indent=4)
