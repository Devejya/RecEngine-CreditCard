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
from pyspark.context import SparkContext
from pyspark.sql import HiveContext, SparkSession
#from ImmutaSparkSession  import ImmutaSparkSession


if __name__ == "__main__":
    #Set this to output the whole dataframe without truncating

    parser = argparse.ArgumentParser(description='Date and time needed for data file')

    parser.add_argument('--date', dest = 'date',  action='store', type=str, required = True)

    results = parser.parse_args()

    #Check if the required files do exist
    if not (os.path.isfile('./'+results.date+'/Data/df_production.h5')):
        print ("Production Set file does NOT exist")
        sys.exit(30)

    if not (os.path.isfile('./'+results.date+'/Model/model_trained.p')):
        print ("Trained Model Pickle File does NOT exist")
        sys.exit(30)

    if not (os.path.isfile('./'+results.date+'/Data/Train_Set.h5')):
        print ("Train Set file does NOT exist")
        sys.exit(30)

    #Train Set
    train_set = pd.read_hdf('./'+results.date+'/Data/Train_Set.h5', 'df')
    y_train, x_train = train_set['match_cpc_after'], train_set.drop('match_cpc_after', axis=1)

    #Load Trained Model
    with open('./'+results.date+'/Model/model_trained.p', 'rb') as fp:
        model = pickle.load(fp)

    #Read Production Data
    prod_data = pd.read_hdf('./'+results.date+'/Data/df_production.h5', 'df')
    x_prod = prod_data.drop(['match_cpc_after', 'tsys_acct_id'], axis=1)
    print('Prod Data: \n', prod_data.head(), '\n', prod_data.shape)

    #Run Model on the data
    y_prod = model.predict(x_prod)
    #pred_cpc = pd.Series(le_encoder.inverse_transform(np.array(y_prod)), name='Predicted_CPC')
    pred_cpc = pd.Series(y_prod, name='Predicted_CPC')
    print('Pred_CPC: \n', pred_cpc.head(), pred_cpc.shape)
    
    #Predicted Probabilities (2D Array [[],[]])
    predict_test_prob = model.predict_proba(x_prod)
    #Corresponding Classes
    model_classes = model.classes_
    print(model_classes)
    
    #Setting up spark sessions, etc. to be able to send data to HDFS (Hive)
    # Creating a SparkContext
    sc = SparkContext(appName="Prem_Model_Run")
    # Optional creation of a HiveContext
    sql_context = HiveContext(sc)
    # Optional creation of a SparkSession
    spark = SparkSession(sc)
    spark = (SparkSession.builder.enableHiveSupport().getOrCreate())
    
    #Save Model results (Currently commented out since the CSV will be too large to save in git repo)
    prod_data_results = prod_data.drop('match_cpc_after', axis=1).reset_index(drop=True)
    #model_results = pd.concat([prod_data_results, pred_cpc], axis=1)
    #model_results.to_csv('./'+results.date+'/Production/model_predictions.csv')
    #print('model_results: \n', model_results.head())

    #Convert pandas dataframe to spark df & Save model results to HDFS (Hive)
    #model_results_spark = spark.createDataFrame(model_results)
    #model_results_spark.write.format('hive').mode("append").saveAsTable("anp_camktedw1_sandbox.jai_Premiumization_Model_Results");
    
    #Feature Importance
    #init shap explainer and create summary plot
    explainer = shap.TreeExplainer(model, x_train)
    shap_values = explainer.shap_values(x_prod[:1],approximate=True)
    print(shap_values)
    shap.summary_plot(shap_values, x_prod[:1], class_names = model_classes, show=True)
    plt.savefig('./'+results.date+'/Production/summary_plot.pdf')
#plot_type="bar"
    #save shap_values to CSV file along with predicted values and Tsys_acct_ID
    final_dataframe = pd.concat([prod_data['tsys_acct_id'], x_prod, shap_values, pred_cpc], axis=1)
    #final_dataframe.to_csv('./'+results.date+'/Production/shap_and_predicted.csv')
    print(final_dataframe.head())
    final_dataframe_spark = spark.createDataFrame(final_dataframe)
    final_dataframe_spark.write.format('hive').mode("overwrite").saveAsTable("anp_camktedw1_sandbox.jai_Premiumization_Model_Results_SHAP");
