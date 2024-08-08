## baseline: random (uniform and task's label distribution), zero rule (predict majority class)

#%%

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score

import os, random

random.seed(0)

#%%

task_name = 'loan_approval_prediction' ## loan_approval_prediction or review_priority_prediction

target_cols = {
    'loan_approval_prediction': 'is_approved',
    'review_priority_prediction': 'is_high_priority'
}


target_col = target_cols[task_name]

train_data = pd.read_csv('../dataset/cleaned/{}/train_original_data.csv'.format(task_name))
test_data = pd.read_csv('../dataset/cleaned/{}/test_original_data.csv'.format(task_name))

y = train_data[target_col]
train_data = train_data.drop(target_col, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(train_data, y, test_size=0.15, shuffle=False)


n_pos = np.sum(y_train)
n_neg = len(y_train) - n_pos

pos_chance = n_pos/len(y_train)
neg_chance = n_neg/len(y_train)

n_test = len(test_data)
y_test = test_data[target_col]
y_test = y_test.astype(int)

#%%

## predict probability
prob = list(np.random.random(n_test))

#%%

## predict majority class
major_prediction = [1.0] * n_test

# %%

labels = [0.0,1.0]

## predict from uniform distribution
uniform_dist_pred = random.choices(labels, [0.5,0.5],k=n_test)

# %%

## predict from label distribution (get distribution from training data)
label_dist_pred = random.choices(labels, [neg_chance, pos_chance],k=n_test)

# %%

data_imbalance_methods = ['imb-data', 'class-weight', 'SMOTE', 'Tomek']

roc_auc = roc_auc_score(y_test, prob)

def eval_result(baseline_name, pred):
    result = classification_report(y_test, pred, output_dict=True)

    result_rows = []

    for k,v in result.items():
        data_row = {}

        if k in ['0','1', 'macro avg']:
            data_row['class'] = k
            for met, val in v.items():
                if met != 'support':
                    data_row[met] = round(val,2)
            result_rows.append(data_row)

    result_df = pd.DataFrame()
    result_df['data-imbalanced-handling'] = data_imbalance_methods
    result_df['model'] = baseline_name
    

    ## store result of each class
    result_all_class_df = pd.DataFrame(result_rows)
    result_all_class_df['model'] = baseline_name

    result_all_class_df = result_df.merge(result_all_class_df, on='model', how = 'inner')
    
    mcc = matthews_corrcoef(y_test, pred)

    result_single_val_df = pd.DataFrame()
    result_single_val_df['data-imbalanced-handling'] = data_imbalance_methods
    result_single_val_df['model'] = baseline_name
    result_single_val_df['AUC'] = roc_auc
    result_single_val_df['MCC'] = mcc

    result_single_val_df = result_single_val_df[['model', 'data-imbalanced-handling', 'AUC', 'MCC']]
    
    return result_all_class_df, result_single_val_df

#%%

result_all_class_df_major_pred, result_single_val_df_major_pred = eval_result('major-class', major_prediction)

# %%

result_all_class_df_uniform_label_pred, result_single_val_df_uniform_label_pred = eval_result('uniform-label', uniform_dist_pred)
# %%

result_all_class_df_label_dist_pred, result_single_val_df_label_dist_pred = eval_result('label-dist', label_dist_pred)
# %%

result_all_class_df = pd.concat([result_all_class_df_major_pred, result_all_class_df_uniform_label_pred, result_all_class_df_label_dist_pred])

result_single_val_df = pd.concat([result_single_val_df_major_pred, result_single_val_df_uniform_label_pred, result_single_val_df_label_dist_pred])

# %%

result_all_class_df.to_csv('../result/eval_metrics/{}_each_class_baseline.csv'.format(task_name), index=False)
result_single_val_df.to_csv('../result/eval_metrics/{}_all_classes_baseline.csv'.format(task_name), index=False)
# %%
