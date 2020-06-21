#!/bin/bash
now="$(date)"
now="$(date +'%d_%m_%Y_%I_%M')"
printf "Current date in dd/mm/yyyy format %s\n" "$now"

mkdir "$now"
mkdir "$now"/Model
mkdir "$now"/Data
mkdir "$now"/Production

#cp ../df_transformed_fixed.h5 "$now"/Data
cp ../df_cat_no_tgc.h5 "$now"/Data

#pip install -U -r requirements.txt
pip install sklearn
pip install hyperopt
pip install xgboost
pip install tables
pip install h2o

FILE=/$now/df_cat_no_tgc.h5
if [ -f "$FILE" ]; then
    echo "$FILE exist"
fi

printf "Step 3: HyperParameter Optimisation\n"
printf "python hyperparam_opt_akora_h2o_XGB.py --date $now\n"
python ./hyperparam_opt_akora_h2o_XGB.py --date $now
if [ $? -eq 0 ]
then
  echo "Successfully executed pythonhyperparam_opt"
else
  # Redirect stdout from echo command to stderr.
  echo "Script exited with error. Data File not found" >&2
  exit $?
fi
# Exit Script if Data File not found in python script
printf "HyperParameter Tuning finished running\n"
printf "============================\n"

git add .
git commit -a -m "HyperParam Tuning Finished $now"
git push

printf "Step 4: Model Training\n"
printf "model_train_akora.py --date $now\n"
python ./model_train_akora.py --date $now
if [ $? -eq 0 ]
then
  echo "Successfully executed Model Training"
else
  # Redirect stdout from echo command to stderr.
  echo "Script exited with error. Data File not found" >&2
  exit $?
fi
# Exit Script if Data File not found in python script
printf "Model Training finished \n"
printf "============================\n"

git add .
git commit -a -m "Model Training Finished $now"
git push


printf "Step 5: Model Run\n"
printf "model_run_akora.py --date $now\n"
python ./model_run_akora.py --date $now
if [ $? -eq 0 ]
then
  echo "Successfully executed model_run_akora"
else
  # Redirect stdout from echo command to stderr.
  echo "Script exited with error. Data File not found" >&2
  exit $?
fi
# Exit Script if Data File not found in python script
printf "model_run_akora finished running\n"
printf "============================\n"

git add .
git commit -a -m "Model Run Finished $now"
git push

#Send final data to Hive
python data_to_hiveql.py --f ./$now/Production/shap_and_predicted.csv --db anp_camktedw1_sandbox --table jai_prem_final_$now --txt hiveql.txt

printf "HiveQL txt file created"

#Use Beeline_heat and no_hub to send data to HEAT

