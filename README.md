# Premiumization

<h2>Enhanced Right Carding</h2>
Premiumization project incorporates a comprehensive approach to all sort of proactive premium or upsell migrations.

# Project Workflow

## Step 1: Data Collection:

* This is done by SQL stored procedures in JoinedQuery.sql
* It's Aggregated and stored as a CSV.
* Due to limitations regarding the number of columns which can be pivoted in SQL Server, That part is done in SAS

## Step 2: Data Transformation:

* Data once collected is aggregated (Avg for the three months prior to the migration made by the user).
* Production dataset (Not used for training or testing) is tagged.
Gamers are removed.
* CPC_Before and After are defined and CPC_Before is one hot encoded.
* All Categorical features are converted into either binary(dichotomous) or numerical (Continous) features.
The data is efficiently stored as an HDF5 file.

## Step 3: Correlation Analysis:

* The Pearson correlation coefficient is calculated between all the features.
* This works since all the features are either continous or dichotomous.
* For dichotomous variables, point-biserial corr is used. But it is theoretically equal to pearson correlation.
Thus, if more categorical features which are ordinal or non dichotmous are added,
chi squared test or kendalls or spearmanns correlation will be required.

* One of the correlated features is removed.
* The default threshold is 0.85 but can be changed using the --coeff flag when running the script

* Finally, the final dataframe is stored efficiently as an HDF5 file.

## Step 3: HyperParameter Optimisation:

* Production dataset is seperated and saved as an HDF5 file.
* TrainTest split is performed.
* Label is encoded and LabelEncoder is saved as pkl file to be used later for inverse transform.
* Test set and Train set are also saved as HDF5 files for model training (Next Step).
* parameter space is defined using hyperopt.
* Objective function is defined and optimised using 3fold CV using the metric ROC-AUC.(Subject to change)
* Best Score and Best parameters are saved for model training.

## Step 4: Model Training:

* Test Set, Training Set and Best Parameters are loaded and the XGB model is trained and fit.
* Test results are saved. (Classification report)
* Trained model is saved as a pkl file.

## Step 5: Model Run on Production Set:

* Production data is loaded.
* Trained model is run on production data.
* Production data and the predictions made are saved as hdf5 file.
* Feature importance is calculated using SHAP.
* Save SHAP summary plot as pdf.
* Save prod data, precicted y and shap values as CSV.

## Move data to HEAT

* <strong>data_to_hiveql.py</strong> can be used to create a text file which will have the HiveQL stored procedure to move the given data to hive.
* This will create a new table. ie, drop previous table if exists then create new table and insert values from dataframe
* The text file then needs to be moved to an EDGENODE server and ran using
```bash
nohup beeline_heat -f <filepath>
```
In the edge node terminal

### Steps to Transfer Data
* 1) Use <strong>data_to_hive.py</strong> to turn your data to HiveQL scripts using the following command in your conda or shell terminal.
```bash
py <path>/data_to_hiveql.py --f <Data File Path> --db <DataBase Name> --table <Table Name> --txt <Text File Name>
```
Example,
```bash
py data_to_hiveql.py --f C:\Users\RAGHUD6\premiumization_feb\Data\ORD.csv --db sandbox --table jai_prem_ORD_CSV --txt hiveql_ORD.txt
```

The Text Files with HiveQL scripts will be written to the same folder as the data_to_hiveql.py.

The files will look like the following using my example,
hiveql_ORD.txt (This file will have the creation of table in it)
hiveql_ORD1.txt
hiveql_ORD2.txt (Files with INSERT Statements with data)
... etc.

* 2) Transfer the Text files with HiveQL scripts to the EDGENODE using WINSCP

* 3) Use Putty or MobaXterm to get access to the terminal of the EDGENODE server (Same as the one used in step 2)

* 4) Check if the files exist in the directory (type ls in the terminal and press enter)

* 5) Use the nohup beeline_heat command mentioned above to run all the files

* 6) You can even make a bash file to do so

## Batch file info (Windows):
The batch file is used to control the flow and provide required arguments to all the python scripts in the process.

## Bash file info (Unix (Mac/Linux)):
The bash file is used to control the flow and provide required arguments to all the python scripts in the process.
Same as batch file but for UNIX OS.
