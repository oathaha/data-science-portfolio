# Credit-Risk-Analysis

## Overview

This directory contains script to run experiment for predicting status/review priority of loan application. 
* For the status of loan application, I built machine learning models to predict whether a loan application should be accepted or refused. 
* For the review priority of loan application, I built machine learning models to predict whether a loan application has high priority or low priority.

More details can be found in this [link](https://sites.google.com/view/chanathip-pornprasit/data-science-portfolio/loan-application-statuspriority-prediction).

## How to Replicate the Experimental Results

To replicate the experimental results, please follow the steps below
1. Get the raw dataset from this [link](https://zenodo.org/records/13787952) and put it to `dataset/raw/`.
2. Go to `./script/` directory
3. Run `clean-data.ipynb` to clean the raw dataset.
4. Run `prepare-data.py` to prepare data for model training.
5. If you want to explore the cleaned dataset, you can run code in `visualize-data-loan-grant-prediction.ipynb` and `visualize-data-review-priority-prediction.ipynb`. 
6. Run `select_important_features.py` to select top-10 features for training models.
7. Run `train-model.py` to train models. For this script, there are 3 arguments
		`--task`: the task to train models (`loan-app-pred` or `priority-pred`)
		`--use_selected_feature`: this flag indicates that models are trained by using selected features in 6.
		`--handle_imb_data`: how to handle imbalanced dataset.
				
	| `--handle_imb_data` | description |
	|--|--|
	| leave blank | do nothing (train models with original data) |
	| over-sampling | use SMOTE to rebalance data |
	|  under-sampling | use Tomek to rebalance data |
	| weight | use balanced class weight to train models |



8. Run `get_result_from_baselines.py` to get the results from baselines (i.e., major class prediction, uniform prediction, class distribution prediction).
9. Run `evaluate.py` to get evaluation results.
10. Run `show-evaluation-results.ipynb` to see evaluation results.

## Experimental Setup

In the experiment, I use grid search to search for the best hyper-parameters for all models. The details of hyper-parameters used for grid search are as follows.

|Model| Hyper-Parameter Search Space | Note
|--|--| -- |
| Decision Tree | criterion = ['gini', 'entropy'] <br> min_samples_split = [2, 3, 5, 7] <br> min_samples_leaf = [1, 3, 5, 7] | -- |
| Logistic Regression | C = [1.0, 10.0, 100.0] <br> solver = ['lbfgs', 'liblinear', 'newton-cg'] | -- |
| K-nearest Neighbors | n_neighbors = [3, 5, 7, 10] <br> power = [1,2] | -- |
| Random Forest | min_samples_split = [2, 3, 5, 7] <br> min_samples_leaf = [1, 3, 5, 7] <br> n_estimator = [50, 100, 300] | -- |
| Gradient Boosting | l2_regularization = [0, 0.3, 0.5, 1.0] <br> learning_rate = [0.1, 0.3, 0.5, 1.0] | -- |
| Extreme Gradient Boosting | n_estimator = [50, 100, 300] <br> learning_rate_ada = [0.1, 0.5, 1.0, 5.0] | -- |
| AdaBoost + Decision Tree  <br> AdaBoost + Logistic Regression| n_estimator = [10, 50, 100] <br> learning_rate = [0.1, 0.5, 1.0, 5.0] <br> | use the best hyper-parameter of Decision Tree and Logistic Regression to initialize base model |
| Bagging + Logistic Regression | n_estimator = [10, 50, 100] | use the best hyper-parameter of Logistic Regression to initialize base model  |

## Result

### Result of application priority prediction

- For the low-priority class, the majority of models that are trained with imbalanced data and data rebalanced by Tomek achieve higher precision and F1 when compared to the models that are trained by using SMOTE and class-weight to handle data imbalanced problem. In contrast, the majority of models that are trained by using SMOTE and class-weight to handle data imbalanced problem achieve higher recall when compared to the models that are trained with imbalanced data and data rebalanced by Tomek.
- For the high-priority class, all models achieve similar performance regardless of data imbalanced handling techniques, and achieve higher performance compared to the results of the low-priority class.
- The overall performance (macro-average) shows that majority of models that are trained with imbalanced data and data rebalanced by Tomek achieve higher precision and F1, while majority of models that are trained by using SMOTE and class-weight to handle data imbalanced problem achieve higher recall.


### Result of approval status prediction
- For the refused class, the majority of models that are trained with imbalanced data and data rebalanced by Tomek achieve higher precision and F1 when compared to the models that are trained by using SMOTE and class-weight to handle data-imbalanced problems. In contrast, the majority of models that are trained by using SMOTE and class-weight to handle data imbalanced problem achieve higher recall when compared to the models that are trained with imbalanced data and data rebalanced by Tomek.
- For the approved class, the majority of models that are trained by using SMOTE and class-weight to handle data imbalanced problem achieve higher precision than the models that are trained with imbalanced data and data rebalanced by Tomek. In contrast, the majority of models that are trained with imbalanced data and data rebalanced by Tomek achieve higher recall when compared to the models that are trained by using SMOTE and class-weight to handle data imbalanced problem. On the other hand, the majority of models that are trained with imbalanced data, and data rebalanced by Tomek and SMOTE achieve higher F1 than the models that are trained by using class-weight to handle data imbalanced problem.
- The overall performance (macro-average) shows that majority of models that are trained with imbalanced data and data rebalanced by Tomek achieve higher precision, while the majority of models that are trained by using SMOTE and class-weight to handle data-imbalanced problems achieve higher recall. On the other hand, the majority of models achieve similar performance regardless of data-imbalanced handling techniques.