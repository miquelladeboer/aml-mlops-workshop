# Quickstart file

## Set up Machine Learning workspace
Navigate to `infrastructue/scripts/create_mlworkspace.py`
* set azure configs

```python
# Create the workspace using the specified parameters
ws = Workspace.create(
    name='<name of workspace>',
    subscription_id='<subscription ID>',
    resource_group='<resource group name>',
    location='<location>',
    create_resource_group=True,
    sku='basic',
    exist_ok=True,
    auth=AzureCliAuthentication()
)
```

* run python script

## Set up data local
Navigate to `data_movement_helpers/load_data_from_web.py`
* run python script

## Set up Data Exploration
Navigate to `code/data exploration/data_exploration.ipyb`
* Run python script

## Set up Data engineering local
Navigate to code/data validation and preparation/data_engineering.py
* At the top of the file, set local path to your local path (Not relative as azureML cannot output local)
* Run script
* You now have prepared data local 

Navigate to `code/data validation and preparation/data_engineering_submit.py`

Make sure the settings are:

```python
data_local = "yes" # allowed options: "yes", "no"
subset = "yes"  # allowed options: "yes", "no"
```

* Run python script

## Set up local training
Navigate to `code/modeling/train.py`
* Run python script 

Navigate to `code/modeling/train_submit.py`

Make sure settings are:

```python
models = 'randomforest'
data_local = True
subset = True
hyperdrive = False
```

* Run python script

## Set up train many models local
Make sure settings are:

```python
models = 'sklearnmodels'
data_local = True
subset = True
hyperdrive = False
```

* Run python scripts
* Inspect local run

## Move data to the cloud
Navigate to `data_movement_helpers/load_data_to_cloud.py`
* Run file

Navigate to `data_movement_helpers/define_dataset_raw.py`

Set current week number correct in line 6:

```python
startWeek = 19  # put here the current week for the first time
```

## Data preperation in the cloud
Navigate to `environments/data_preperation_subset/create_runconfig_data_preparation_yaml.py`

* Run script

Navigate to `environments/data_preperation/create_runconfig_data_preparation_yaml.py`

* Run script

Navigate to `code/data validation and preparation/data_engineering_submit.py`

Set params:

```python
data_local = "no"  # allowed options: "yes", "no"
subset = "yes"  # allowed options: "yes", "no"
```

* Run script

set params:

```python
data_local = "no"  # allowed options: "yes", "no"
subset = "no"  # allowed options: "yes", "no"
```
* Run script

Navigate to `data_movement_helpers/define_dataset_prepared.py`

Set current week number correct in line 6:

```python
startWeek = 19  # put here the current week for the first time
```

* Run script

## Train model with sklearn in the cloud
Navigate to `environments/sklearn_full/create_runconfig_sklearn_yaml.py`

* Run script

Navigate to `environments/sklearn_subsetl/create_runconfig_sklearn_yaml.py`

* Run script

Navigate to `code/modeling/train_submit.py`

Set params:

```python
models = 'randomforest'
data_local = False
# if data_local is true, subset is alwats true
subset = True
# hyperdrive only works with deeplearning
hyperdrive = False
```

* Run script

## Train model with deeplearning on PyTorch in the cloud
set params:

```python
models = 'deeplearning'
data_local = False
# if data_local is true, subset is alwats true
subset = True
# hyperdrive only works with deeplearning
hyperdrive = False
```

* Run script

## Hyperdrive
Set params:

```python
models = 'deeplearning'
data_local = False
# if data_local is true, subset is alwats true
subset = True
# hyperdrive only works with deeplearning
hyperdrive = True
```

## Data validation
navigate to `code/data exploration/create_baseline_profile.py`

* Run script

navigate to `environments/data_validation/create_runconfig_data_validation_yaml.py`

* Run script

navigate to `environments/data_validation_subset/create_runconfig_data_validation_yaml.py`

* Run script

navigate to `code/data validation and preparation/data_validation_submit.py`

set parameters:

```python
subset = "yes"
```

* Run script

set parameters:

```python
subset = "no"
```

* Run Script

## create historic profile

navigate to `environments/data_profiling/create_runcnfig_data_profiling_yaml.py`

* run script

navigate to `code/data validation and preparation/create_historic_profile_submit.py`

set params:

```python
newprofile = 'yes'
```

## create historic profile over time
Navigate to `data_movement_helpers/load_data_to_cloud.py`

In line 10 change:

```python
weekNumber = str(date.today().isocalendar()[1])
```
 to
 ```python
weekNumber = str(date.today().isocalendar()[1]+1)
```

* Run script

navigate to `data_movement_helpers/define_dataset_raw.py`

in line 7 change:

```python
weekNumber = (date.today().isocalendar()[1])
```
to

```python
weekNumber = (date.today().isocalendar()[1]+1)
```

* Run script

navigate to `code/data validation and preparation/create_historic_proflie.py`

in line 102 change:

```python
today = str(date.today())
```
to
```python
today = str(date.today() + + timedelta(days=7))
```

navigate to c`ode/data validation and preparation/create_historic_profile_sumbit.py`

in line 12 change:

```python
newprofile = 'yes'
```
to
```python
newprofile = 'no'
```

* Run script

Repeat many times.

## BUILDING THE PIPELINE
Navigate to `code/pipelines/retrain_model.py`

* Run script
