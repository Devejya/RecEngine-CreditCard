import pandas as pd 
import numpy as np 
from xgboost import XGBClassifier
import json
import os.path
import sys
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import jaccard_score
try:
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle
    
    
def metric_per_CPC_before(cpc_before, true_cpc_after, pred_cpc_after, prod):
    '''
    return a dataframe with the metrics (Accuracy and F1) per CPC Before
    input: pandas series, pandas series, pandas series (CPC_Before, True_CPC_After, Pred_CPC_After)
    output: Pandas Dataframe
    effect: prints dataframe
    '''
    #Init the metrics dataframe list
    metrics_df = pd.DataFrame(columns = ['CPC_Before','Accuracy (Weighted Jaccard Score)','F1 Weighted'])
    #concatenate the series into a dataframe
    cpc_df = pd.concat([cpc_before, pred_cpc_after, true_cpc_after], axis=1, ignore_index=True)
    cpc_df.columns = ['cpc_before','cpc_after_pred', 'cpc_after_true']
    
    #create a list of CPCs before and iterate through them
    list_cpc = list(set(cpc_before.unique()))
    metrics_df['CPC_Before'] = list_cpc
    jaccard_scores = []
    f1_scores = []
    
    for cpc in list_cpc:
        cpc_before_current = cpc_df[cpc_df['cpc_before']==cpc]
        jaccard_scores.append(jaccard_score(cpc_before_current['cpc_after_true'], cpc_before_current['cpc_after_pred'], average='weighted'))
        f1_scores.append(f1_score(cpc_before_current['cpc_after_true'], cpc_before_current['cpc_after_pred'], average='weighted'))
        
    metrics_df['Accuracy (Weighted Jaccard Score)'] = jaccard_scores
    metrics_df['F1 Weighted'] = f1_scores
    
    print("Metrics DataFrame (Per CPC Before): \n",metrics_df.head())
    
    return metrics_df
         
    
if __name__ == "__main__":
    #Set this to output the whole dataframe without truncating
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    parser = argparse.ArgumentParser(description='Data File needed')

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)
    
    parser.add_argument('--prod', action='store_true', default=False,
                        dest='prod',
                        help='Set to run in production. ie, will not print debug statements')

    results = parser.parse_args()

    #Check if the required files do exist
    if not (os.path.isfile('./'+results.date+'/Data/Test_Set.h5')):
        print ("Test Set file does NOT exist")
        sys.exit(40)

    if not (os.path.isfile('./'+results.date+'/Data/Train_Set.h5')):
        print ("Train Set file does NOT exist")
        sys.exit(40)

    if not (os.path.isfile('./'+results.date+'/Data/Best_Params.json')):
        print ("Best Parameters file does NOT exist")
        sys.exit(40)

    #Read Test, Train Set and Best HyperParameters
    
    test_set = pd.read_hdf('./'+results.date+'/Data/Test_Set.h5', 'df')
    y_test, x_test = test_set['match_cpc_after'], test_set.drop('match_cpc_after', axis=1)
    train_set = pd.read_hdf('./'+results.date+'/Data/Train_Set.h5', 'df')
    y_train, x_train = train_set['match_cpc_after'], train_set.drop('match_cpc_after', axis=1)

    with open('./'+results.date+'/Data/Best_Params.json', 'r') as fp:
        best_params = json.load(fp)

    #Initialize the model with the best parameters and fit the model on training data
    #DMatrices not required with the new XGB library
    model = XGBClassifier(best_params)
    print('Best Params: ', best_params)
    
    import time
    start = time.time()
    print('Fitting the Model')
    
    model.fit(x_train, y_train)
    
    print('Model Done Fitting')
    print("Model Training took %.2f seconds" % ((time.time() - start)))
    
    #Storing model as pickle file
    print('Storing model as pickle file\n')
    with open('./'+results.date+'/Model/model_trained.p', 'wb') as fp:
        pickle.dump(model, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #Test the model
    predict_test = model.predict(x_test)
    y_actual=y_test.reset_index(drop=True)
    
    if not results.prod:
        print('pred_y: ', predict_test)
        print('y_test type: ', type(y_test))
        print('pred_y series: ', pd.Series(predict_test).head())
        print('y_test series: ', y_actual.head())
        y_true_pred = pd.concat([pd.Series(predict_test), y_actual],axis=1, ignore_index=True)
        print('Y_TRUE: ', y_true_pred.head())
        y_true_pred.columns = ['Predicted CPC','Actual CPC']
        print('Predict_Test: ', y_true_pred.head(5))
    
    #Predicted Probabilities
    predict_test_prob = model.predict_proba(x_test)
    
    if not results.prod:
        print('Predict_Proba: ', predict_test_prob)

    # Check the f1-scores for all target classes
    print('Classification Report and Confusion Matrix: \n')
    print(classification_report(y_test,predict_test))
    print(confusion_matrix(y_test,predict_test))
    
    #Reverse One Hot Encoding
    cpc_before = x_test.filter(regex='cpc_before').idxmax(axis=1).str.replace('match_cpc_before_','').str.upper().reset_index(drop=True)
    #Check the metrics for each CPC Before
    metrics_df = metric_per_CPC_before(cpc_before, y_actual, pd.Series(predict_test, name='Predicted_CPC'), results.prod)

    #Store model results
    print('Storing model results\n')
    report = classification_report(y_test, predict_test, output_dict=True)
    df_results = pd.DataFrame(report).transpose()
    df_results.to_csv('./'+results.date+'/Model/Classification_report.csv')
    metrics_df.to_csv('./'+results.date+'/Model/Metrics_Per_CPC.csv',index=False)
    conf_matrix_df = pd.DataFrame(confusion_matrix(y_test,predict_test), columns = model.classes_, index = model.classes_)
    conf_matrix_df.to_csv('./'+results.date+'/Model/Confusion_Matrix.csv')
