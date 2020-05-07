# Demo Setup

Quickstart file:
navigate to infrastructue/scripts/create_mlworkspace.py
- set azure configs:
- run python script

DATA LOCAL
Navigate to data_movement_helpers/load_data_from_web.py
- run python script

Date exploration:
Navigate to code/data exploration/data_exploration.ipyb
-	Run python scripts
Data engineering LOCAL
Navigate to code/data validation and preparation/data_engineering.py
-	At the top of the file, set local path to your local path (azureML cannot outputto local)
-	Run script
-	You now have prepared data local 
We can perform the same step using azure for logging 
Navigate to code/data validation and preparation/data_engineering_submit.py
Make sure the settings are:
data_local = "yes" # allowed options: "yes", "no"
subset = "yes"  # allowed options: "yes", "no"

-	Run python script
-	Inspect run in aml
TRAIN LOCAL
Open code/modeling/train.py
-	Run python script 
Open code/modeling/train_submit.py
Make sure settings are:
models = 'randomforest'
data_local = True
subset = True
hyperdrive = False

-	Run python scripts
-	Inspect local run

TRAIN MANY MODELS LOCAL

models = 'sklearnmodels'
data_local = True
subset = True
hyperdrive = False

-	Run python scripts
-	Inspect local run
MOVE TO THE CLOUD
Hyperdrive -> cloud compute + data to cloud + environments
Compute is already available in set-up
Move data to the cloud  raw data from current week to cloud
Navigate to data_movement_helpers/load_data_to_cloud.py
-	Run file
Define dataset raw data
Navigate to data_movement_helpers/define_dataset_raw.py
Set current week number correct in line 6:
startWeek = 19  # put here the current week for the first time

Now that we have pur datasets, we can update the ID’s in out config files we need for data preparation
Navigate to environments/data_preperation_subset/create_runconfig_data_preparation_yaml.py
Run script
Navigate to environments/data_preperation/create_runconfig_data_preparation_yaml.py
Run script

We need prepared data – data engineering in cloud
Navigate to code/data validation and preparation/data_engineering_submit.py
Set params:
data_local = "no"  # allowed options: "yes", "no"
subset = "yes"  # allowed options: "yes", "no"

run file

set params
data_local = "no"  # allowed options: "yes", "no"
subset = "no"  # allowed options: "yes", "no"


run file
Navigate to data_movement_helpers/define_dataset_prepared.py

Set current week number correct in line 6:
startWeek = 19  # put here the current week for the first time

Run pyhton file

SKLEARN IN THE CLOUD
Navigate to environments/sklearn_full/create_runconfig_sklearn_yaml.py
Run file
Navigate to environments/sklearn_subsetl/create_runconfig_sklearn_yaml.py
Run file
Navigate to code/modeling/train_submit.py 
Set paras
models = 'randomforest'
data_local = False
# if data_local is true, subset is alwats true
subset = True
# hyperdrive only works with deeplearning
hyperdrive = False

run script
models = 'deeplearning'
data_local = False
# if data_local is true, subset is alwats true
subset = True
# hyperdrive only works with deeplearning
hyperdrive = False

run script
HYPERDRIVE ON REMOTE COMPUTE
models = 'deeplearning'
data_local = False
# if data_local is true, subset is alwats true
subset = True
# hyperdrive only works with deeplearning
hyperdrive = True

data validation
navigate to code/data exploration/create_baseline_profile.py
run script

navigate to environments/data_validation/create_runconfig_data_validation_yaml.py
run script

navigate to environments/data_validation_subset/create_runconfig_data_validation_yaml.py
run script

navigate to code/data validation and preparation/data_validation_submit.py
set parameter

subset = "yes"
run script

set parameter

subset = "no"
create historic profile

navigate to environments/data_profiling/create_runcnfig_data_profiling_yaml.py
run scripts

navigate to code/data validation and preparation/create_historic_profile_submit.py

set params
newprofile = 'yes'

CREATE PROFILE OVER TIME
Navigate to data_movement_helpers/load_data_to_cloud.py
In line 10 change
weekNumber = str(date.today().isocalendar()[1])
 to
weekNumber = str(date.today().isocalendar()[1]+1)

run script
navigate to data_movement_helpers/define_dataset_raw.py
in line 7 change
weekNumber = (date.today().isocalendar()[1])
to
weekNumber = (date.today().isocalendar()[1]+1)

run script
navigate to code/data validation and preparation/create_historic_proflie.py
in line 102 change
today = str(date.today())

to
today = str(date.today() + + timedelta(days=7))

navigate to code/data validation and preparation/create_historic_profile_sumbit.py
in line 12 change
newprofile = 'yes'

to
newprofile = 'no'

run script

repeat many times.

BUILDING THE PIPELINE
Navigate to code/pipelines/retrain_model.py
Run script
