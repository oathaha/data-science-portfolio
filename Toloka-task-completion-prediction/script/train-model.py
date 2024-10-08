#%%
import os, pickle, json, argparse

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import  AdaBoostRegressor, BaggingRegressor

#%%

parser = argparse.ArgumentParser()
parser.add_argument('--use_selected_feature', action = 'store_true')

args = parser.parse_args()

use_selected_feature = args.use_selected_feature

# %%
## Prepare dataset

if use_selected_feature:
    df = pd.read_csv('../dataset/cleaned/train_processed_selected_features.csv')
    model_subdir = 'use_selected_features'
else:
    df = pd.read_csv('../dataset/cleaned/train_processed_df.csv')
    model_subdir = 'use_all_features'

print('load data finished')
print('-'*30)

df = df.reset_index()

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.15, shuffle=False)

y = df['completion-time-in-minutes']
x = df.drop(['completion-time-in-minutes', 'index'], axis=1)

print('prepare data finished')
print('-'*30)

del df


# %%

## Define hyper-parameters for model fine-tuning with grid search

random_state = 0

alpha = [1, 5, 10]
solver = ['sparse_cg']
l1_ratio = [0.3, 0.5, 0.7, 1.0]
max_iter = [100, 500, 1000]
n_estimator = [10, 50, 100]
learning_rate = [0.1, 0.5, 1.0, 5.0]
loss_func = ['linear', 'square']


search_params = {
    'ridge': {
        'alpha': alpha,
        'max_iter': max_iter
    },
    'lasso': {
        'alpha': alpha,
        'max_iter': max_iter
    },
    'elasticNet': {
        'alpha': alpha,
        'max_iter': max_iter,
        'l1_ratio': l1_ratio
    },
    'adaboost': {
        'n_estimators': n_estimator,
        'learning_rate': learning_rate,
        'loss': loss_func
    },
    'bagging': {
        'n_estimators': n_estimator
    }
}


#%%

## initialize regression model

model_dir = '../model/{}'.format(model_subdir)
os.makedirs(model_dir, exist_ok=True)

linear = LinearRegression(n_jobs=32)
ridge = Ridge(random_state=random_state)
lasso = Lasso(random_state=random_state)
elasticNet = ElasticNet(random_state=random_state)


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

def grid_search_reg_model(model, params):
    model_name = type(model).__name__

    print_training_info(model_name, params)

    gs = GridSearchCV(
        model,
        param_grid=params,
        scoring='neg_root_mean_squared_error',
        n_jobs=4,
        cv=[(train_idx, test_idx)],
        verbose = True
    )

    gs.fit(x, y)

    print('train model done')
    print('saving results')

    log_artifacts(gs)


#%%

## fit simple regression model here

# print('training a regression model')
# print('-'*30)

# linear.fit(x.loc[train_idx], y.loc[train_idx])
# log_artifacts(linear, log_grid_search_result=False)

# print('finished training')
# print('-'*30)

## grid search for single regression models

# grid_search_reg_model(lasso, search_params['lasso'])
# grid_search_reg_model(elasticNet, search_params['elasticNet'])
# grid_search_reg_model(ridge, search_params['ridge'])

# %%

def get_best_params_from_base_model(base_model_name):
    model_dir = '../model/{}/{}'.format(model_subdir, base_model_name)

    with open(os.path.join(model_dir, 'best_params.json')) as f:
        best_params = json.load(f)

    return best_params


def grid_search_ensemble_model(base_model_name, ensemble_model_name, params=None):

    if base_model_name not in ['Lasso', 'Ridge', 'ElasticNet', 'Linear']:
        print('wrong base model name')
        exit(0)

    if ensemble_model_name not in ['adaboost', 'bagging']:
        print('wrong ensemble model name')
        exit(0)

    if base_model_name in ['Lasso', 'Ridge', 'ElasticNet']:

        try:
            best_params = get_best_params_from_base_model(base_model_name)
        except:
            print('please train the base {} model first'.format(base_model_name))
            exit(0)

    if base_model_name == 'Lasso':
        base_model = Lasso(
            alpha = best_params['alpha'], max_iter=best_params['max_iter']
        )
    elif base_model_name == 'Ridge':
        base_model = Ridge(
            alpha = best_params['alpha'], max_iter=best_params['max_iter']
        )
    elif base_model_name == 'ElasticNet':
        base_model = ElasticNet(
            alpha = best_params['alpha'], max_iter=best_params['max_iter'],
            l1_ratio=best_params['l1_ratio']
        )
    elif base_model_name == 'Linear':
        base_model = LinearRegression(n_jobs=4)

    if ensemble_model_name == 'adaboost':
        model = AdaBoostRegressor(estimator=base_model, random_state=random_state)

    elif ensemble_model_name == 'bagging':
        model = BaggingRegressor(estimator=base_model, random_state=random_state)

    model_name = ensemble_model_name + '_' + base_model_name

    print_training_info(model_name, params)

    gs = GridSearchCV(
        model,
        param_grid=params,
        scoring='neg_root_mean_squared_error',
        n_jobs=4,
        cv=[(train_idx, test_idx)],
        verbose=1
    )

    gs.fit(x, y)

    print('train model done')
    print('saving results')

    log_artifacts(gs, is_ensemble=True)
    
#%%

grid_search_ensemble_model('Linear', 'adaboost', params=search_params['adaboost'])

grid_search_ensemble_model('Lasso', 'adaboost', params=search_params['adaboost'])
grid_search_ensemble_model('ElasticNet', 'adaboost', params=search_params['adaboost'])
grid_search_ensemble_model('Ridge', 'adaboost', params=search_params['adaboost'])


grid_search_ensemble_model('Lasso', 'bagging', search_params['bagging'])
grid_search_ensemble_model('ElasticNet', 'bagging', search_params['bagging'])
grid_search_ensemble_model('Ridge', 'bagging', search_params['bagging'])
grid_search_ensemble_model('Linear', 'bagging', search_params['bagging'])
# %%
