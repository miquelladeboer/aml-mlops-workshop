## Lab 1: Introduction ##
In this lab, we will learn how to do data exploration on the data. The dataset used in this example is the 20 newsgroups dataset. It will be automatically downloaded, then cached. The newsgroup datasets contains text documents that are classified into 20 categories. In this solution we will only use 4 of the categories. In this lab, we are going to do 2 things:
1. load data from the web to local computer
2. explore the data

# Pre-requirements #
1. If not familiair with the concepts of data exploration, read the comprehensive guide on [Data Exploration](https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration)
2. Finished the setup file [02_infrastructure](https://github.com/miquelladeboer/aml-mlops-workshop/blob/master/labs/02_infrastructure.md)

# Load datas from web to lcoal computer #
Before we can analyse the data, we will first load the data from the web (the 20newsgroup http form sklear) into a local file on our computer. Later in the labs, we will learn how to move data to cloub (blob storage) and work with data in the cloud.

1. Navigate to `data_movement_helpers/load_data_from_web.py`

    This file will help us with loadin data from the web into a local folder. We are going to inspect the file step by step before excecuting it.

2. Definee the outputs folder

    First step is to define the outputs folder. This will be the folder where the outputs data will be. In this example we put the data into the folder `outputs/raw_data`.

    ```python
    # Set outputrs folder to outputs/raw_data
    OUTPUTSFOLDER = "outputs/raw_data"

    # create outputs folder if not exists, create the folder
    if not os.path.exists(OUTPUTSFOLDER):
        os.makedirs(OUTPUTSFOLDER)
    ```

3.  Select categories

    We then select the categories we want to use. In our example, we chose 4 categories, but you can choose more/less or different categories if you like.

    ```python
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    ```

4. Load data from web

    We are now going to load the data from the 20newsgroup sklearn, where we select a random sample from the http based on `subset`, that can either be `train` or `test`.

    ```python
    newsgroupdata = fetch_20newsgroups(
        subset=data_split,
        categories=categories,
        shuffle=True,
        random_state=42
    )
    ```

5. 