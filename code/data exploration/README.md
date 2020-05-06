# Data Exploration
Before starting an ML experiment, we need of course to explore and validate data. As I have already mentioned, we can use notebooks, even within vscode, to perform our data exploration. 

An very important part of MLOps is data validation and data validation over time. We want to make sure if we automate the process of re-training that our data is not drifting (or at least we want to be aware of drifts) and that the data is not biased. More basic validation like if the format of the data is still as expected or if there are any missing values present are also important to validate. 

We understand why bias is bad for model training, but I think it is important to point out why data drifting is important to be aware of. Of course, data drifting is not necessary a bad thing. It is something that we are expecting. That is also part of the reason why we retrain our model occasionally, to reflect the new incoming data as well. However, what I see at most customers, is that the have model validation after they trained the model. Within model validation, customers often check if certain parameter are within a certain bandwidth. When this happens, the model validation will fail. However, as we can imagine, if the data changes a lot, these parameters are expected to go out of bandwidth. As I do recommend to let you model fail when this happens, it is recommended to have a data profiling report ready that can explain these failures.

Therefore, I would recommend to build a baseline profile of basic and more advanced statistics of your dataset, that you found when exploring the data, and every time you retrain your model, validate if your new data is within a certain interval of these statistics. I would recommend building a small report of warning and output this to the blob storage. I would also recommend to build a small dataset of all these statistics over time. This way, you can you Power BI for example, to create reports and profiles of your data and check and guarantee data quality and transparency. 

This folder contains some scripts that are intended for data exploration and the create of a baseline profile of your data. In this script you will find two files.

* data_exploration.ipynb

    Note here that we are able to open Jupyter notebooks from vscode. This file contains some data explorations. Including, data types, check for imbalenced/biased classes, check language of text messages and basic text statistics like word cound, sentiment analyses and tfidf. 

* create_baseline_profile.py

    This file is used to create a baseline profile for the most important statistics in the dataset, inclusing:
    - mean of classes
    - standard deviation of classes
    - average number of stop words
    - average length of words
    - average sentiment of dataset
    - word count profile

    We use these statistics to validate new data as it comes in. When new data comes in, we can check if the statistics of the new data are within a 5 or 10 percent interval of the baseline statistics. In this way, we can be warned that data might be drifting, classes become biased or that people try to manipulate the algorithm on purpose. The pyhton file for data validation can be found [here](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/code/data%20validation%20and%20preparation/data_validation.py). The creation of this baseline profile is static and will only be updated every month or when needed manually, to ensure the check of data sanity.



