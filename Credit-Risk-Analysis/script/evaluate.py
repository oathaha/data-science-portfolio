
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

task_name = 'loan_approval_prediction'
target_col = 'is_approved'

df = pd.read_csv('../dataset/cleaned/{}/test_processed_data.csv'.format(task_name))

y = df[target_col]
x = df.drop([target_col], axis=1)


# %%

## Get evaluation results

def eval_result(model_name, imb_data_handling_method):

    model_dir = '../model/{}/{}/{}/'.format(task_name, imb_data_handling_method, model_name)

    print('loading model from', model_dir)

    with open(model_dir + 'model.pkl', 'rb') as f:
        model = pickle.load(f)

    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]

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

    ## store result of all classes
    roc_auc = roc_auc_score(y, prob)
    mcc = matthews_corrcoef(y, pred)

    result_dict = {
        'model': model_name,
        'data-imbalanced-handling': imb_data_handling_method,
        'AUC': roc_auc,
        'MCC': mcc
    }

    return result_all_class_df, result_dict

# %%
result_dir = '../result/'

os.makedirs(result_dir, exist_ok=True)

## just for testing (change to real value later...)
model_names = ['DecisionTreeClassifier']
data_imb_handling_methods = ['imb-data']

alL_result_each_class_df = []
all_result_all_classes_rows = []

for model_name in model_names:
    for method in data_imb_handling_methods:
        result_all_class_df, result_dict = eval_result(model_name, method)

        alL_result_each_class_df.append(result_all_class_df)
        all_result_all_classes_rows.append(result_dict)  

# %%

result_each_class = pd.concat(alL_result_df)
result_all_classes = pd.DataFrame(all_result_rows)

result_each_class.to_csv(os.path.join(result_dir, '{}_result_each_class.csv'.format(task_name)), index=False)
result_all_classes.to_csv(os.path.join(result_dir, '{}_result_all_classes.csv'.format(task_name)), index=False)

#%%
