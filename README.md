# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

As a scope of this project, we are tasked to create and optimize ML pipelineusing the Python SDK for which, a custom-coded standard **[Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)** model is provided.<br/>Utilizing knowledge acquired through this course, we need to optimize Hyperparametes using **[HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py)** package of Azue Python SDK. Result of this method needs to be compared with a model built and optimized using **[Azure AutoML](https://azure.microsoft.com/en-ca/services/machine-learning/automatedml/)** on the same dataset.

### Project Workflow Steps:
![Alt Text](https://github.com/Panth-Shah/AzureML_Optimize_MachineLearning_Pipeline_in_Azure/blob/master/Run_Results/creating-and-optimizing-an-ml-pipeline.png)
*Source: Machine Learning Engineer with Microsoft Azure Nanodegree Program by Udactiy*

## Summary

### Problem Statement

In this project, we demonstate how to tain a logistic regression model using Azue ML Python SDK with Scikit-learn to perform classification on the [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) dataset. The classification goal is to predict if the client will subscribe to a term deposite or not with the bank.

Next step would be to further improve the accuracy by automating optimization process our model's hyperparameters `C` and `max-itr` using Azure Machine Learning's hyperparameter tuning package HyperDrive.

There may be a better Machine Learning algoithm to use for this application, which will be determined using Azure ML's Automated ML feature, which will simplify model building pocess and will create a high quality trained model using provided dataset.

Final result will be determined comparing performance metrics of models obtained from both the approaches.

### Input variables:

Bank Client Data:<br/>
1 - `age`: (numeric)<br/>
2 - `job`: type of job (categorical:    'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br/>
3 - `marital`: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br/>
4 - `education`: (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br/>
5 - `default`: has credit in default? (categorical: 'no','yes','unknown')<br/>
6 - `housing`: has housing loan? (categorical: 'no','yes','unknown')<br/>
7 - `loan`: has personal loan? (categorical: 'no','yes','unknown')<br/>

Related with the last contact of the current campaign:<br/>
8 - `contact`: contact communication type (categorical: 'cellular','telephone')<br/>
9 - `month`: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')<br/>
10 - `day_of_week`: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br/>
11 - `duration`: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.<br/>

Other Attributes:<br/>
12 - `campaign`: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br/>
13 - `pdays`: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br/>
14 - `previous`: number of contacts performed before this campaign and for this client (numeric)<br/>
15 - `poutcome`: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br/>

Social and Economic Context Attributes:<br/>
16 - `emp.var.rate`: employment variation rate - quarterly indicator (numeric)<br/>
17 - `cons.price.idx`: consumer price index - monthly indicator (numeric)<br/>
18 - `cons.conf.idx`: consumer confidence index - monthly indicator (numeric)<br/>
19 - `euribor3m`: euribor 3 month rate - daily indicator (numeric)<br/>
20 - `nr.employed`: number of employees - quarterly indicator (numeric)<br/>

### Output variable (desired target):

21 - `y`: has the client subscribed a term deposit? (binary: 'yes','no')

### Solution Result Explanation:

- For Scikit-learn Logistic Regression model, **90.83%** accuracy was achived after tuning hyperparameters with the help of `HyperDrive` package.

- Using `Automated ML` for the same dataset, **91.68%** accuracy was achived with best performing model being `Voting Ensemble`.

***Dataset Source**: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014*

## Scikit-learn Pipeline

### Pipeline Architecture

- Initialize Workspace and Create an Experiment:

	This step will initialize an existing workspace object using `get` method of `Workspace` class.
	New experiment is created to track all the runs in this workspace for Scikit-learn Logistic Regression model.
	
- Create AmlCompute:

	A compute target (Azure ML managed compute) is created to train ML model. Type of Compute Cluster is created based on configuration (`vm_size`, `max_node`) prvided.
	
- Prepare training script to train classification model:

	For this project, training script `train.py` was already provided. Script includes
	- Load data from csv file by creating TabularDataset using `TabularDatasetFactory`
	- Perform data pre-preprocessing(clean up & transformation) task
	- Split data into train and test sets 
	- Calling `sklearn.linear_model.LogisticRegression` providing Inverse of regularization strength (`C`) & Maximum number of iterations (`max_itr`) hyperparameters.
	
- Configure Training Run:

	Using `ScriptRunConfig` package, job-specific information such as training script `train.py`, target environment, and compute resources to use are configured to submit with the training job.

- Hyperparameter Tuning:

	`HyperDriveConfig` object is created to use for this purpose, which includes information about hyperparameter space sampling, termination policy, primary metric, estimator, run config, maximum number of concurrent runs, maximum total number of runs to create and maximum duration of the HyperDrive run.
	
- Submit and Monitor Hyperparameter Tuning job:

	We will submit the hyperdrive run to the experiment created for SKlearn model training purpose.<br/> 
	To monitor progress of model training including properties, logs, and metrics from Jupyter notebook, we will use AzureML widget `RunDetails`.
	
- Capture Best Run for HyperDrive:

	Using `get_best_run_by_primary_metric` method of HyperDriveRun class, we will capture best performing run amongst all child runs. This is identified based on primary metric parameter (`Accuracy`) specified in the HyperDriveConfig.
	
- Register and Save Best Model for HyperDrive:

	Upon identifying best performing run having highest accuracy, we can register this model under the workspace for deployment using `register_model` method or save into local repository using `download_file` of AzureML core Run class.
	
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
Following command is executed to delete compute cluster created

```
cpu_cluster.delete()
```

![Alt Text](https://github.com/Panth-Shah/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Run_Results/DeleteCluster.JPG)
