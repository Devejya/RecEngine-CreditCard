#!/bin/bash
now="$(date)"
now="$(date +'%d_%m_%Y_%I_%M')"
printf "Current date in dd/mm/yyyy format %s\n" "$now"

mkdir "$now"
mkdir "$now"/Model
mkdir "$now"/Data
mkdir "$now"/Production

#cp ../df_transformed_fixed.h5 "$now"/Data
#cp ../df_transformed_prod.h5 "$now"/Data

#pip install -U -r requirements.txt
pip install sklearn
pip install hyperopt
pip install xgboost
pip install tables

FILE=/$now/df_transformed.h5
if [ -f "$FILE" ]; then
    echo "$FILE exist"
fi

#printf "Step 1: Data Transformation\n"
#printf "python datatransform.py --date $now run with -h to see all options \n"
#python ./datatransform.py -keep-aero --date $now 
#if [ $? -eq 0 ]
#then
#  echo "Successfully executed datatransform"
#else
  # Redirect stdout from echo command to stderr.
#  echo "Script exited with error. Data File not found" >&2
#  exit $?
#fi
# Exit Script if Data File not found in python script
#printf "DataTransform finished running\n"
#printf "============================\n"


printf "Step 2: Correlation Analysis \n"
printf "python correlation_analysis.py --date $now\n"
python ./correlation_analysis.py --date $now --coeff 0.9
if [ $? -eq 0 ]
then
  echo "Successfully executed Correlation Analysis"
else
  # Redirect stdout from echo command to stderr.
  echo "Script exited with error. Data File not found" >&2
  exit $?
fi
# Exit Script if Data File not found in python script
printf "Correlation Analysis finished running\n"
printf "============================\n"


printf "Step 3: HyperParameter Optimisation\n"
printf "python hyperparam_opt_akora.py --date $now\n"
python ./hyperparam_opt_akora.py --date $now
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
