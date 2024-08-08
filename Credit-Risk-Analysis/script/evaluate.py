
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef

import pandas as pd

import pickle, os



# %%

## Read test set

target_cols = {
    'loan_approval_prediction': 'is_approved',
    'review_priority_prediction': 'is_high_priority'
}

task_name = 'review_priority_prediction'
target_col = target_cols[task_name]

df = pd.read_csv('../dataset/cleaned/{}/test_processed_data.csv'.format(task_name))

y = df[target_col]
x = df.drop([target_col], axis=1)


# %%

## Get evaluation results

## class order num is 0, 1
col_names = {
    'loan_approval_prediction': ['Refused', 'Approved'],
    'review_priority_prediction': ['Low-Priority', 'High-Priority']
}

def eval_result(model_name, imb_data_handling_method):

    model_dir = '../model/{}/{}/{}/'.format(task_name, imb_data_handling_method, model_name)

    print('loading model from', model_dir)

    with open(model_dir + 'model.pkl', 'rb') as f:
        model = pickle.load(f)

    pred = model.predict(x)
    prob = model.predict_proba(x)

    result = classification_report(y, pred, output_dict=True)

    result_rows = []

    # print(result)

    for k,v in result.items():
        data_row = {
            'model': model_name,
            'data-imbalanced-handling': imb_data_handling_method
        }

        if k in ['0.0','1.0', 'macro avg']:
            data_row['class'] = k
            for met, val in v.items():
                if met != 'support':
                    data_row[met] = round(val,2)
            result_rows.append(data_row)

    ## store result of each class
    result_all_class_df = pd.DataFrame(result_rows)

    ## store result of all classes (roc_auc is for positive label only)
    roc_auc = roc_auc_score(y, prob[:, 1])
    mcc = matthews_corrcoef(y, pred)

    result_dict = {
        'model': model_name,
        'data-imbalanced-handling': imb_data_handling_method,
        'AUC': roc_auc,
        'MCC': mcc
    }


    prob_df = pd.DataFrame(prob, columns = col_names[task_name])
    prob_df['model_name'] = model_name

    return result_all_class_df, result_dict, prob_df

# %%
base_result_dir = '../result'
result_dir = os.path.join(base_result_dir, 'eval_metrics')
prob_dir = os.path.join(base_result_dir, 'prob_values', task_name)

os.makedirs(result_dir, exist_ok=True)
os.makedirs(prob_dir, exist_ok=True)

## just for testing (change to real value later...)
model_names = ['DecisionTreeClassifier', 'LogisticRegression']
data_imb_handling_methods = ['imb-data', 'Tomek']

alL_result_each_class_df = []
all_result_all_classes_rows = []
prob_val = {s:[] for s in data_imb_handling_methods}


for model_name in model_names:
    for method in data_imb_handling_methods:
        result_all_class_df, result_dict, prob_df = eval_result(model_name, method)

        alL_result_each_class_df.append(result_all_class_df)
        all_result_all_classes_rows.append(result_dict)
        prob_val[method].append(prob_df)

# %%

result_each_class = pd.concat(alL_result_each_class_df)
result_all_classes = pd.DataFrame(all_result_all_classes_rows)

result_each_class.to_csv(os.path.join(result_dir, '{}_result_each_class.csv'.format(task_name)), index=False)
result_all_classes.to_csv(os.path.join(result_dir, '{}_result_all_classes.csv'.format(task_name)), index=False)

for data_imb_handling_method, df_list in prob_val.items():
    final_df = pd.concat(df_list)
    final_df.to_csv(os.path.join(prob_dir, '{}_prob.csv'.format( data_imb_handling_method)), index=False)

#%%

