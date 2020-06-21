import pandas as pd 
import numpy as np 
from xgboost import XGBClassifier
import json
import os.path
import sys
import argparse
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle


if __name__ == "__main__":
    #Set this to output the whole dataframe without truncating
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    parser = argparse.ArgumentParser(description='Data File needed')

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)

    results = parser.parse_args()

    #Check if the required files do exist
    if not (os.path.isfile('../'+results.date+'/Data/Test_Set.h5')):
        print ("Test Set file does NOT exist")
        sys.exit(40)

    if not (os.path.isfile('../'+results.date+'/Data/Train_Set.h5')):
        print ("Train Set file does NOT exist")
        sys.exit(40)

    if not (os.path.isfile('../'+results.date+'/Data/Best_Params.json')):
        print ("Best Parameters file does NOT exist")
        sys.exit(40)

    #Read Test, Train Set and Best HyperParameters
    test_set = pd.read_hdf('../'+results.date+'/Data/Test_Set.h5', 'df')
    y_test, x_test = test_set['Match_CPC_After'.lower()], test_set.drop('Match_CPC_After'.lower(), axis=1)
    train_set = pd.read_hdf('../'+results.date+'/Data/Train_Set.h5', 'df')
    y_train, x_train = train_set['Match_CPC_After'.lower()], train_set.drop('Match_CPC_After'.lower(), axis=1)

    with open('../'+results.date+'/Data/Best_Params.json', 'r') as fp:
        best_params = json.load(fp)

    #Initialize the model with the best parameters and fit the model on training data
    #DMatrices not required with the new XGB library
    model = xgb.XGBClassifier(best_params)
    model.fit(x_train, y_train)

    #Test the model
    predict_test = model.predict(x_test, pred_contribs = True)

    # Check the f1-scores for each group
    print('Classification Report and Cofusion Matrix: \n')
    print(classification_report(y_test,predict_test4))
    print(confusion_matrix(y_test,predict_test4))

    #Store model results
    print('Storing model results\n')
    report = classification_report(y_test, predict_test4, output_dict=True)
    df_results = pandas.DataFrame(report).transpose()
    df_results.to_csv('../'+results.date+'/Model/Classification_report.csv')

    #Storing model as pkl file
    print('Storing model as pickle file\n')
    with open('../'+results.date+'/Model/model_trained.p', 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)
