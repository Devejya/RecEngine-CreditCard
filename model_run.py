import pandas as pd 
import numpy as np 
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
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

    parser = argparse.ArgumentParser(description='Date and time needed for data file')

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)

    results = parser.parse_args()

    #Check if the required files do exist
    if not (os.path.isfile('../'+results.date+'/Data/df_production.h5')):
        print ("Production Set file does NOT exist")
        sys.exit(30)

    if not (os.path.isfile('../'+results.date+'/Model/model_trained.p')):
        print ("Trained Model Pickle File does NOT exist")
        sys.exit(30)

    if not (os.path.isfile('../'+results.date+'/Data/Train_Set.h5')):
        print ("Train Set file does NOT exist")
        sys.exit(30)

    if not (os.path.isfile('../'+results.date+'/Data/Target_encoder.p')):
        print ("Label Encoder file does NOT exist")
        sys.exit(30)

    #load the encoder file
    pkl_file = open('../'+results.date+'/Data/Target_encoder.p', 'rb')
    le_encode = pickle.load(pkl_file) 
    pkl_file.close()

    #Train Set
    train_set = pd.read_hdf('../'+results.date+'/Data/Train_Set.h5', 'df')
    y_train, x_train = train_set['Match_CPC_After'.lower()], train_set.drop('Match_CPC_After'.lower(), axis=1)

    #Load Trained Model
    with open('../'+results.date+'/Model/model_trained.p', 'rb') as fp:
        model = pickle.load(fp)

    #Read Production Data
    prod_data = pd.read_hdf('../'+results.date+'/Data/df_production.h5', 'df')
    x_prod = prod_data.drop(['Match_CPC_After'.lower(), 'Tsys_Acct_ID'.lower()], axis=1)

    #Run Model on the data
    y_prod = model.predict(x_prod, pred_contribs = True)
    pred_cpc = pd.Series(le_encoder.inverse_transform(np.array(y_prod)), name='Predicted_CPC')

    #Save Model results
    model_results = pd.concat([prod_data.drop('Match_CPC_After'.lower(), axis=1), pred_cpc], axis=1)
    model_results.to_csv('../'+results.date+'/Production/model_predictions.csv')

    #Feature Importance
    #init shap explainer and create summary plot
    explainer = shap.TreeExplainer(model, x_train)
    shap_values = explainer.shap_values(x_prod,approximate=True)
    shap.summary_plot(shap_values, x_prod, plot_type="bar", class_names = ['TAC', 'TAI', 'TAW', 'TFC', 'TGC', 'TGS', 'TIC', 'TPB', 'TPT'], show=False)
    plt.savefig('../'+results.date+'/Production/summary_plot.pdf')

    #save shap_values to CSV file along with predicted values and Tsys_acct_ID
    final_dataframe = pd.concat([prod_data['Tsys_Acct_ID'.lower()], x_prod, shap_values, pred_cpc], axis=1)
    final_dataframe.to_csv('../'+results.date+'/Production/shap_and_predicted.csv')
