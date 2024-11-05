#%%
import os, pickle, json, argparse

import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

## from https://xgboost.readthedocs.io/en/stable/get_started.html
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


#%%

parser = argparse.ArgumentParser()
parser.add_argument('--task', type = str, required=True)
parser.add_argument('--handle_imb_data', type=str, default='', required=False)

args = parser.parse_args()

task = args.task
handle_imb_data = args.handle_imb_data
use_selected_features = args.use_selected_features


# %%
## Prepare dataset

if task == 'loan-app-pred':
    data_subdir = 'loan_approval_prediction'
    model_subdir = 'loan_approval_prediction'
    target_col = 'is_approved'

elif task == 'priority-pred':
    data_subdir = 'review_priority_prediction'
    model_subdir = 'review_priority_prediction'
    target_col = 'is_high_priority'

else:
    print('wrong task name')
    print('task name must be "loan-app-pred" or "priority-pred"')
    exit(0)

    
train_df_dir = '../dataset/cleaned/{}/train_processed_selected_features.csv'.format(data_subdir)


print('load training data from', train_df_dir)

df = pd.read_csv(train_df_dir)

print('load data finished')
print('-'*30)

df = df.reset_index()

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.15, shuffle=False)

random_state = 0

#%%

## handle imbalanced data

def split_x_and_y(df, target_col):
    y = df[target_col]
    x = df.drop([target_col, 'index'], axis=1)

    return x, y

class_weight = None
sample_weight = None

if handle_imb_data in ['', 'weight']:
    x, y = split_x_and_y(df, target_col)
    
    if handle_imb_data == '':
        print('train model with original dataset')
        imb_handling_method = 'imb-data'
        

    elif handle_imb_data == 'weight':
        print('use balanced class weight to train model')
        imb_handling_method = 'class-weight'
        class_weight = 'balanced'

        ## compute sample weight for xgboost
        ## only compute for training set
        ## all samples in validation set have the same weight (i.e., 1)
        train_df = df.iloc[train_idx]
        valid_df = df.iloc[test_idx]

        x_train, y_train = split_x_and_y(train_df, target_col)
        x_valid, y_valid = split_x_and_y(valid_df, target_col)

        sample_weight_train = compute_sample_weight(class_weight='balanced', y = y_train)
        sample_weight_valid = np.array([1]*len(y_valid))

        sample_weight = np.concatenate((sample_weight_train, sample_weight_valid), axis=0)

        print(sample_weight)

        del x_train, y_train, x_valid, y_valid

    
elif handle_imb_data in ['over-sampling', 'under-sampling']:

    train_df = df.iloc[train_idx]
    valid_df = df.iloc[test_idx]

    x_train, y_train = split_x_and_y(train_df, target_col)
    x_valid, y_valid = split_x_and_y(valid_df, target_col)

    if handle_imb_data == 'over-sampling':
        print('use SMOTE to rebalance data')
        imb_handling_method = 'SMOTE'
        resampler = SMOTE(random_state=random_state)

    elif handle_imb_data == 'under-sampling':
        print('use Tomek to rebalance data')
        imb_handling_method = 'Tomek'
        resampler = TomekLinks()

    x_train, y_train = resampler.fit_resample(x_train, y_train)

    x = pd.concat([x_train, x_valid])
    y = pd.concat([y_train, y_valid])

    ## reset index for indicing in grid search
    x = x.reset_index().drop('index', axis=1)

    train_idx = np.arange(len(x_train))
    test_idx = np.arange(len(x_train), len(x))

    print('train idx:', train_idx)
    print('valid idx:', test_idx)

print('prepare data finished')
print('-'*30)

del df

# %%

## Define hyper-parameters for model fine-tuning with grid search

C = [1.0, 10.0, 100.0]
# kernel = ['linear', 'poly', 'rbf']
solver = ['lbfgs', 'liblinear', 'newton-cg']
criterion = ['gini', 'entropy'] 
min_samples_split = [2, 3, 5, 7]
min_samples_leaf = [1, 3, 5, 7]
n_estimator = [50, 100, 300]
n_neighbors = [3, 5, 7, 10]
power = [1,2]
learning_rate = [0.1, 0.3, 0.5, 1.0]
l2_regularization = [0, 0.3, 0.5, 1.0]
learning_rate_ada = [0.1, 0.5, 1.0, 5.0]

search_params = {
    'decision-tree': {
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'criterion': criterion
    },
    'SVM':{
        'C': C,
        # 'kernel': kernel
    },
    'Logistic-regression': {
        'C': C,
        'solver': solver
    },
    'KNN': {
        'n_neighbors': n_neighbors,
        'p': power
    },
    'random-forest':
    {
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'n_estimators': n_estimator
    },
    'gradient-boosting':{
        'learning_rate': learning_rate,
        'l2_regularization': l2_regularization
    },
    'xgboost': {
        'n_estimators': n_estimator,
        'learning_rate': learning_rate
    },
    'adaboost': {
        'n_estimators': n_estimator,
        'learning_rate': learning_rate_ada
    },
    'bagging': {
        'n_estimators': n_estimator
    }
}


#%%

model_dir = '../model/{}/{}'.format(model_subdir, imb_handling_method)

print('create directory {} to store models'.format(model_dir))

os.makedirs(model_dir, exist_ok=True)


## initialize classification models

decision_tree = DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)
knn = KNeighborsClassifier()
lr = LogisticRegression(random_state=random_state, class_weight=class_weight)
gbt = HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight)
xgb = XGBClassifier(random_state=random_state)
rf = RandomForestClassifier(random_state=random_state, class_weight=class_weight)

def log_artifacts(model_obj, log_grid_search_result = True, is_ensemble = False):

    if log_grid_search_result:

        if is_ensemble:
            model_name = type(model_obj.best_estimator_).__name__ + '_' + type(model_obj.best_estimator_.estimator_).__name__
        else:
            model_name = type(model_obj.best_estimator_).__name__

        real_model_dir = os.path.join(model_dir, model_name)

        os.makedirs(real_model_dir, exist_ok=True)

        print('saving artifacts to', real_model_dir)

        with open(os.path.join(real_model_dir, 'model.pkl'),'wb') as f:
            pickle.dump(model_obj.best_estimator_, f)

        with open(os.path.join(real_model_dir, 'best_params.json'), 'w') as f:
            json.dump(model_obj.best_params_, f)

        with open(os.path.join(real_model_dir, 'best_score.txt'), 'w') as f:
            f.write(str(model_obj.best_score_))

    else:
        model_name = type(model_obj).__name__

        real_model_dir = os.path.join(model_dir, model_name)

        os.makedirs(real_model_dir, exist_ok=True)

        print('saving artifacts to', real_model_dir)

        with open(os.path.join(real_model_dir, 'model.pkl'),'wb') as f:
            pickle.dump(model_obj, f)

    print('finished saving artifacts')
    print('-'*30)


def print_training_info(model_name, params):
    print('performing hyper-parameter optimization on', model_name)
    print('hyper-parameter search space:')
    for k,v in params.items():
        print('  {}:\t'.format(k), v)

    print('-'*30)

def grid_search_cls_model(model, params):
    model_name = type(model).__name__

    print_training_info(model_name, params)

    gs = GridSearchCV(
        model,
        param_grid=params,
        scoring='neg_log_loss',
        n_jobs=4,
        cv=[(train_idx, test_idx)],
        verbose = True
    )

    if 'XGB' in model_name:
        gs.fit(x, y, sample_weight = sample_weight)
    else:
        gs.fit(x,y)

    print('train model done')
    print('saving results')

    log_artifacts(gs)


#%%

## grid search for single classification models

grid_search_cls_model(decision_tree, search_params['decision-tree'])
grid_search_cls_model(knn, search_params['KNN'])
grid_search_cls_model(lr, search_params['Logistic-regression'])
grid_search_cls_model(rf, search_params['random-forest'])
grid_search_cls_model(gbt, search_params['gradient-boosting'])
grid_search_cls_model(xgb, search_params['xgboost'])


# %%

def get_best_params_from_base_model(base_model_name):
    real_model_dir = os.path.join(model_dir, base_model_name)

    with open(os.path.join(real_model_dir, 'best_params.json')) as f:
        best_params = json.load(f)

    return best_params


def grid_search_ensemble_model(base_model_name, ensemble_model_name, params=None):

    if base_model_name not in ['LogisticRegression', 'DecisionTreeClassifier']:
        print('wrong base model name')
        exit(0)

    if ensemble_model_name not in ['adaboost', 'bagging']:
        print('wrong ensemble model name')
        exit(0)


    best_params = get_best_params_from_base_model(base_model_name)


    if base_model_name == 'LogisticRegression':
        base_model = LogisticRegression(
            C=best_params['C'], 
            solver=best_params['solver'], 
            random_state=random_state,
            class_weight=class_weight
        )

    elif base_model_name == 'DecisionTreeClassifier':
        base_model = DecisionTreeClassifier(
            min_samples_split=best_params['min_samples_split'], 
            min_samples_leaf=best_params['min_samples_leaf'], 
            criterion=best_params['criterion'], 
            random_state=random_state, 
            class_weight=class_weight
        )
        
    if ensemble_model_name == 'adaboost':
        model = AdaBoostClassifier(estimator=base_model, random_state=random_state)

    elif ensemble_model_name == 'bagging':
        model = BaggingClassifier(estimator=base_model, random_state=random_state)

    model_name = ensemble_model_name + '_' + base_model_name

    print_training_info(model_name, params)

    gs = GridSearchCV(
        model,
        param_grid=params,
        scoring='neg_log_loss',
        n_jobs=4,
        cv=[(train_idx, test_idx)],
        verbose=1
    )

    gs.fit(x, y)

    print('train model done')
    print('saving results')

    log_artifacts(gs, is_ensemble=True)
    
#%%

grid_search_ensemble_model('DecisionTreeClassifier', 'adaboost', params=search_params['adaboost'])
grid_search_ensemble_model('LogisticRegression', 'adaboost', params=search_params['adaboost'])

grid_search_ensemble_model('LogisticRegression', 'bagging', search_params['bagging'])
