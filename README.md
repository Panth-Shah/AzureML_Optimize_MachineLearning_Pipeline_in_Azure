# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.

As a scope of this project, we are tasked to create and optimize ML pipelineusing the Python SDK for which, a custom-coded standard **[Scikit-learn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)** model is provided.<br/>Utilizing knowledge acquired through this course, we need to optimize Hyperparametes using **[HyperDrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py)** package of Azue Python SDK. Result of this method needs to be compared with a model built and optimized using **[Azure AutoML](https://azure.microsoft.com/en-ca/services/machine-learning/automatedml/)** on the same dataset.

### Project Workflow Steps:
![Alt Text](https://github.com/Panth-Shah/AzureML_Optimize_MachineLearning_Pipeline_in_Azure/blob/master/Run_Results/creating-and-optimizing-an-ml-pipeline.png)

Figure 1: Steps to perform to create and optimize ML pipeline

***Source**: Machine Learning Engineer with Microsoft Azure Nanodegree Program by Udactiy*

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

- Using `Automated ML` for the same dataset, **91.70%** accuracy was achived with best performing model being `Voting Ensemble`.

***Dataset Source**: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014*

## Scikit-learn Pipeline

### Pipeline Architecture:

- **Initialize Workspace and Create an Experiment:**

	This step will initialize an existing workspace object using `get()` method of `Workspace` class.
	New experiment is created to track all the runs in this workspace for Scikit-learn Logistic Regression model.
	
- **Create AmlCompute:**

	A compute target (Azure ML managed compute) is created to train ML model. Type of Compute Cluster is created based on configuration (`vm_size`, `max_node`) prvided.
	
- **Prepare training script to train classification model:**

	For this project, training script `train.py` was already provided. Script includes
	- Load data from csv file by creating TabularDataset using `TabularDatasetFactory`
	- Perform data pre-preprocessing(clean up & transformation) task
	- Split data into train and test sets 
	- Calling `sklearn.linear_model.LogisticRegression` providing Inverse of regularization strength (`C`) & Maximum number of iterations (`max_itr`) hyperparameters.
	
- **Configure Training Run:**

	Using `ScriptRunConfig` package, job-specific information such as training script `train.py`, target environment, and compute resources to use are configured to submit with the training job.

- **Hyperparameter Tuning:**

	`HyperDriveConfig` object is created to use for this purpose, which includes information about hyperparameter space sampling, termination policy, primary metric, estimator, run config, maximum number of concurrent runs, maximum total number of runs to create and maximum duration of the HyperDrive run.
	
- **Submit and Monitor Hyperparameter Tuning job:**

	We will submit the hyperdrive run to the experiment created for SKlearn model training purpose.<br/> 
	To monitor progress of model training including properties, logs, and metrics from Jupyter notebook, we will use AzureML widget `RunDetails`.
	
- **Capture Best Run for HyperDrive:**

	Using `get_best_run_by_primary_metric()` method of HyperDriveRun class, we will capture best performing run amongst all child runs. This is identified based on primary metric parameter (`Accuracy`) specified in the HyperDriveConfig.
	
- **Register and Save Best Model for HyperDrive:**

	Upon identifying best performing run having highest accuracy, we can register this model under the workspace for deployment using `register_model()` method or save into local repository using `download_file()` of AzureML core Run class.
	
### Sampling the Hyperparameter Space:

To define random sampling over the search space of hyperparameter we are trying to optimize, we are using AzureML's `RandomParameterSampling` class. Levaraging this method of parameter sampling, users can randomly select hyperparameter from defined search space. With this sampling algorithm, AzureML lets users choose hyperparameter values from a set of discrete values or a distribution over a continuous range. This method also supports early termination of low performance runs, which is a cost effcient approach when training model on aml compute cluster.

Other two approaches supported by AzureML are Grid Sampling and Bayesian Sampling:

	As `Grid Sampling` only supports discrete hypeparameter, it searches over all the possibilities from defined search space. And so more compute resource is required, which is not very budget efficient fo this project. 
	`Bayesian Sampling` method is based on Bayesian optimization algorithm and picks samples based on how previous samples performed to improve the primary metric of new samples. Because of that, more number of runs benefit future selection of samples, which also is not a very cost efficient solution for this project. 

### Advantages of Early Stopping Policy:

While working with Azure's managed aml compute cluster to train classification model for this project, it is important to maintain and imporve computational effciency.Specifying early termination policy autometically terminates poorly performing runs based on configuration parameters (`evaluation_interval`, `delay_evaluation`) provided upon defining these policies. These can be applied to HyperDrive runs and run is cancelled when the criteria of a specified policy are met.

`Bendit Policy`:

Among supported early termination policies by Azure ML, we are using Bendit Policy in this project. This policy is based on slack criteria, and a frequency and delay interval for evaluation. BenditPolicy determines best performing run based on selected primary metric (Accuracy for this project) and sets it as a benchmark to compare it against other runs. `slack_factor`/`slack_amount` configuration parameter is used to specify slack allowed with respect to best performing run. `evaluation_interval` specifies frequency of applying policy and `delay_evaluation` specifies number of intervals to delay policy evaluation for run termination. This parameter ensures protection against premature termination of training run.

Other termination policies supported by Azure ML are `Median Stopping Policy`, `Truncation Selection Policy` & `No Termination Policy`.

![Alt Text](https://github.com/Panth-Shah/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Run_Results/Hyperdrive_AzureStudio_Run.JPG)

Figure 2. Azure ML Studio Experiment submitted with HyperDrive from notebook 

![Alt Text](https://github.com/Panth-Shah/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Run_Results/Hyperdrive_NotebookRun_Accuracy_Plot.JPG)

Figure 3. Plot displaying `Accuracy` obtained from all the child runs in an experiment

![Alt Text](https://github.com/Panth-Shah/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Run_Results/Hyperdrive_NotebookRun_Parameter_Plot.JPG)

Figure 4. Plot displaying `C` and `max-itr` hyperparmeter values selected for all the child runs in an experiment

## AutoML

- Azure Automated Machine Learning (AutoML) prvides capabilities automate iterative task of machine learning model development. In this project, we are using Azure AutoML to train and tune a Machine Learning model to solve classification problem using UCI Bank Marketing dataset with the goal to predict if the client will subscribe to a term deposite with the bank.

### **AutoMLConfig:**

- This class from Azure ML Python SDK represents configuration to submit an automated ML experiment in Azure ML. Configuration parameters used for this project includes
	- `experiment_timeout_minutes`: 30 
	- `task`: classification 
	- `primary_metric`: accuracy
	- `training_data`: Tabular dataset created from csv data file using TabularDatasetFactory
	- `compute_target`: aml-compute cluster
	- `label_column`: y(target)
	- `n_cross_validations`: 5

### **Best Performing Model and Hyperparameters generated by AutoML:**

- Among all the models trained by AutoML, `Voting Ensemble` outperformed all the other models with `91.70% accuracy`.

	- Ensemble models in Automated ML are combination of multiple iterations which may provide better predictions compared to a single iteration and appear as the final iterations of run.
	- Two types of ensemble methods for combining models: **Voting** and **Stacking**
	- Voting ensemble model predicts based on the weighted average of predicted class probabilities.
	- In our project, combined models by Voting ensemble with their selected hyperparameters are as follows. 
	
### Hyperparameters generated for models ensembled in Voting Ensemble:

	prefittedsoftvotingclassifier
	{'estimators': ['0', '1', '18', '17', '23', '4', '21'],
	 'weights': [0.2,
				 0.3333333333333333,
				 0.06666666666666667,
				 0.06666666666666667,
				 0.2,
				 0.06666666666666667,
				 0.06666666666666667]}

	0 - maxabsscaler
	{'copy': True}

	0 - lightgbmclassifier
	{'boosting_type': 'gbdt',
	 'class_weight': None,
	 'colsample_bytree': 1.0,
	 'importance_type': 'split',
	 'learning_rate': 0.1,
	 'max_depth': -1,
	 'min_child_samples': 20,
	 'min_child_weight': 0.001,
	 'min_split_gain': 0.0,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'num_leaves': 31,
	 'objective': None,
	 'random_state': None,
	 'reg_alpha': 0.0,
	 'reg_lambda': 0.0,
	 'silent': True,
	 'subsample': 1.0,
	 'subsample_for_bin': 200000,
	 'subsample_freq': 0,
	 'verbose': -10}

	1 - maxabsscaler
	{'copy': True}

	1 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 1,
	 'gamma': 0,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 3,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'binary:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 1,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 1,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

	18 - sparsenormalizer
	{'copy': True, 'norm': 'max'}

	18 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 1,
	 'eta': 0.3,
	 'gamma': 5,
	 'grow_policy': 'lossguide',
	 'learning_rate': 0.1,
	 'max_bin': 63,
	 'max_delta_step': 0,
	 'max_depth': 10,
	 'max_leaves': 0,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 25,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 1.5625,
	 'reg_lambda': 0.10416666666666667,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 0.7,
	 'tree_method': 'hist',
	 'verbose': -10,
	 'verbosity': 0}

	17 - sparsenormalizer
	{'copy': True, 'norm': 'l1'}

	17 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 0.9,
	 'eta': 0.5,
	 'gamma': 0,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 6,
	 'max_leaves': 0,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 100,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 0.9375,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 0.6,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

	23 - sparsenormalizer
	{'copy': True, 'norm': 'max'}

	23 - xgboostclassifier
	{'base_score': 0.5,
	 'booster': 'gbtree',
	 'colsample_bylevel': 1,
	 'colsample_bynode': 1,
	 'colsample_bytree': 0.5,
	 'eta': 0.5,
	 'gamma': 0.01,
	 'learning_rate': 0.1,
	 'max_delta_step': 0,
	 'max_depth': 10,
	 'max_leaves': 7,
	 'min_child_weight': 1,
	 'missing': nan,
	 'n_estimators': 50,
	 'n_jobs': 1,
	 'nthread': None,
	 'objective': 'reg:logistic',
	 'random_state': 0,
	 'reg_alpha': 0,
	 'reg_lambda': 0.8333333333333334,
	 'scale_pos_weight': 1,
	 'seed': None,
	 'silent': None,
	 'subsample': 1,
	 'tree_method': 'auto',
	 'verbose': -10,
	 'verbosity': 0}

	4 - minmaxscaler
	{'copy': True, 'feature_range': (0, 1)}

	4 - randomforestclassifier
	{'bootstrap': True,
	 'ccp_alpha': 0.0,
	 'class_weight': 'balanced',
	 'criterion': 'gini',
	 'max_depth': None,
	 'max_features': 'log2',
	 'max_leaf_nodes': None,
	 'max_samples': None,
	 'min_impurity_decrease': 0.0,
	 'min_impurity_split': None,
	 'min_samples_leaf': 0.01,
	 'min_samples_split': 0.01,
	 'min_weight_fraction_leaf': 0.0,
	 'n_estimators': 25,
	 'n_jobs': 1,
	 'oob_score': True,
	 'random_state': None,
	 'verbose': 0,
	 'warm_start': False}

	21 - maxabsscaler
	{'copy': True}

	21 - extratreesclassifier
	{'bootstrap': False,
	 'ccp_alpha': 0.0,
	 'class_weight': 'balanced',
	 'criterion': 'gini',
	 'max_depth': None,
	 'max_features': 0.8,
	 'max_leaf_nodes': None,
	 'max_samples': None,
	 'min_impurity_decrease': 0.0,
	 'min_impurity_split': None,
	 'min_samples_leaf': 0.06157894736842105,
	 'min_samples_split': 0.29105263157894734,
	 'min_weight_fraction_leaf': 0.0,
	 'n_estimators': 25,
	 'n_jobs': 1,
	 'oob_score': False,
	 'random_state': None,
	 'verbose': 0,
	 'warm_start': False}

## Pipeline comparison

In this project, we trained **Logistic Regression** model using Azure ML's HyperDrive feature for random sampling search space for hyperparameter tuning and achieved `90.83%` accuracy.

In our second approach, we trained classification model for given dataset using Azure ML's Automated ML feature which was configured to time out after 30 minutes. With this approach, `Voting Ensemble` model has proven best performing with `91.70%`accuracy.

### Architectural Difference:

Logistic Regression is a simple algorithm and easily interpretable. Hyperparameter tuning process used with this project allows us to randomize parameter sampling process, but limitation in number of total runs and number of concurrent runs allowed considering expensive compute resource consumption, we are limited to not explore more discrete values for larger search space and check more permutations of hyperparameters to to find better performing model. 

Whereas, AutomatedML is capable of producing ensemble models which combines results ffrom multiple iterations in a single run to produce better prediction results. However, due to max time limit to run AutoML, accuracy achieved is not significantly higher. With AutomatedML, we don't have to worry about hyperparameter tuning as it autometically does that based on AutoML configuration and so optimum combination of hyperparameters can be obtained.
 
## Future work

- According to Azure AutoML's Data Guardrails analysis, **class immbalance** is detected in the provided dataset for this project. Here, class distribution of sample space in the training dataset is severly disproportionated with non-subscription to subscription instance is 89:11. Because input data has a bias towards one class, this can lead to a falsely perceived positive effect of a model's accuracy.
To improve accuracy of the prediction model, will use synthetic sampling techniques like `SMOTE`, `MSMOTE` and other ensemble techniques to increase the frequency of the minority class or decrease the frequncy of the majority class.

- Avoiding misleading data in order to imporve the performance of our prediction model is a critical step as irrelevant attributes in your dataset can result into overfitting. As a future enhancement of this project, leveraging **Automated ML's Model Interpretability** dashboard, will inspect which dataset feature is essential and used by model to make its predictions and determine best and worst performing features to include/exclude them from future runs. Based on this finding, will customize featurization settings used by Azure AutoML to train our model. `FeaturizationConfig` defines feature engineering configuration for our automated ML experiment, using which we will exclude irrelevent features identified from AutoML's model interpretability dashboard. While training SKLearn Logistic Regression classification model, **[Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)** can also be used to rank the feature and recursively reducing the sets of features.

- With current project, we are using only two hyperparameters `C` and `max-itr` to train Logistic Regression model. Adding additional parameters like `penalty`, `class_weight`, `intercept_scaling`, `n_jobs`, `l1_ratio` etc. will allow us to control training model and performance of our classifier can be improvised. 

- Due to class imbalance problem we have with the given data set, it is possible that model will always only predict class which has higher % instances in the dataset. This results into excellent classification accuracy as it only reflects the underlying class distribution. This situation is called **Accuracy Paradox**, where accuracy is not the best metric to use for performance evaluation of prediction model and can be misleading. As a future improvements of this model, will use additional measures such as **Precision, Recall, F1 Score** etc. to evaluate a trained classifier.

- With more compute resources in hand for future experiments, will perform parameter sampling over the search space of hyperparameter for **HyperDriveConfig** using **Bayesian Sampling** technique. To obtain better results, will go with Azure ML's recommended approach by maximizing number of runs greater than or equal to 20 times the number of hyperparameters being tuned using this sampling technique. Will also keep number of concurrent runs lower, which will lead us to better sampling convergence, since the smaller degree of parallelism increases the number of runs that benefit parameter tuning process by taking reference from previously completed runs.


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
Following command is executed to delete compute cluster created

```
cpu_cluster.delete()
```

![Alt Text](https://github.com/Panth-Shah/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/Run_Results/DeleteCluster.JPG)
