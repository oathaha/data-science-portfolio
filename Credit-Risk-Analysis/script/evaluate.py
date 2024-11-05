
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef

import pandas as pd

import pickle, os, argparse


# %%

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str, required=True)

args = parser.parse_args()


task = args.task
use_selected_features = args.use_selected_features

if task == 'loan-app-pred':
    data_subdir = 'loan_approval_prediction'
    model_subdir = 'loan_approval_prediction'
    target_col = 'is_approved'

elif task == 'priority-pred':
    data_subdir = 'review_priority_prediction'
    model_subdir = 'review_priority_prediction'
    target_col = 'is_high_priority'


test_df_dir = '../dataset/cleaned/{}/test_processed_selected_features.csv'.format(data_subdir)


## Read test set

print('read test data from', test_df_dir)

df = pd.read_csv(test_df_dir)

y = df[target_col]
x = df.drop([target_col], axis=1)


# %%

## Get evaluation results

## class order num is 0, 1
col_names = {
    'loan_approval_prediction': ['Refused', 'Approved'],
    'review_priority_prediction': ['Low-Priority', 'High-Priority']
}

def load_model(model_name, imb_data_handling_method):
    model_dir = '../model/{}/{}/{}/'.format(model_subdir, imb_data_handling_method, model_name)

    print('loading model from', model_dir)

    with open(model_dir + 'model.pkl', 'rb') as f:
        model = pickle.load(f)

    return model


## prob is for positive class only
def evaluate(model_name, imb_data_handling_method, pred, prob):
    result = classification_report(y, pred, output_dict=True)

    result_rows = []

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
base_result_dir = '../result'
result_dir = os.path.join(base_result_dir, 'eval_metrics')
prob_dir = os.path.join(base_result_dir, 'prob_values', data_subdir)

os.makedirs(result_dir, exist_ok=True)
os.makedirs(prob_dir, exist_ok=True)


model_names = [
    "AdaBoostClassifier_DecisionTreeClassifier", 
    "KNeighborsClassifier", 
    "AdaBoostClassifier_LogisticRegression", 
    "LogisticRegression", 
    "BaggingClassifier_LogisticRegression", 
    "RandomForestClassifier", 
    "DecisionTreeClassifier", 
    "XGBClassifier", 
    "HistGradientBoostingClassifier"
]
data_imb_handling_methods = ['imb-data', 'Tomek', 'class-weight', 'SMOTE']

alL_result_each_class_df = []
all_result_all_classes_rows = []
prob_val = {s:[] for s in data_imb_handling_methods}


for method in data_imb_handling_methods:
    pred_all_models_df = pd.DataFrame()
    prob_all_models_df = pd.DataFrame()
    
    ## get results from each model
    for model_name in model_names:
        model = load_model(model_name, method)

        pred = model.predict(x)
        prob = model.predict_proba(x)[:,1]

        pred_all_models_df[model_name] = pred
        prob_all_models_df[model_name] = prob

        result_all_class_df, result_dict = evaluate(model_name, method, pred, prob)

        alL_result_each_class_df.append(result_all_class_df)
        all_result_all_classes_rows.append(result_dict)
        
    ## for ensemble voting
    final_prediction = pred_all_models_df.mode(axis='columns')
    final_prob = prob_all_models_df.mean(axis='columns')

    result_all_class_df, result_dict = evaluate('Ensemble Voting', method, final_prediction, final_prob)

    alL_result_each_class_df.append(result_all_class_df)
    all_result_all_classes_rows.append(result_dict)
        

# %%

result_each_class = pd.concat(alL_result_each_class_df)
result_all_classes = pd.DataFrame(all_result_all_classes_rows)

result_each_class.to_csv(os.path.join(result_dir, '{}_result_each_class.csv'.format(data_subdir)), index=False)
result_all_classes.to_csv(os.path.join(result_dir, '{}_result_all_classes.csv'.format(data_subdir)), index=False)

