#%%

import pandas as pd

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from scipy.stats import pointbiserialr


import os, argparse
#%%

## task_name: loan_approval_prediction or review_priority_prediction
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, required=True)

args = parser.parse_args()

base_dir = '../dataset/cleaned/'

task_name = args.task_name

labels_str = {
    'loan_approval_prediction': 'is_approved', 
    'review_priority_prediction': 'is_high_priority'
}

if task_name not in list(labels_str.keys()):
    print('wrong task name')
    print('task name must be in', list(labels_str.keys()))
    exit(0)

label_str = labels_str[task_name]

#%%

train_df = pd.read_csv(os.path.join(base_dir,task_name, 'train_original_data.csv'))
test_df = pd.read_csv(os.path.join(base_dir,task_name, 'test_original_data.csv'))

original_train_df = train_df.copy()

#%%

num_cols = [
    'AMT_ANNUITY', 'AMT_APPLICATION',
    'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE', 'RATE_DOWN_PAYMENT',
    'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED',
    'CNT_PAYMENT'
]

cat_cols = list(set(list(train_df.columns)) - set(num_cols) - set([label_str]))

labels = train_df[label_str]

#%%

## encode categorical variables presented in string to numerical values (e.g., 1,2,3) for later steps

for c in cat_cols:

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[c])
    train_df[c] = label_encoder.transform(train_df[c])

# %%

## use chi-square statistics to obtain top-5 features 
feature_selector = SelectKBest(chi2, k=5)
feature_selector.fit(train_df[cat_cols], labels)

top_5_cat_feature_cols = feature_selector.get_feature_names_out().tolist()


# %%

## calculate Point Biserial to measure the correlation between numerical variables and predicted classes

test_results = []

for c in num_cols:

    test_result = pointbiserialr(labels, train_df[c])

    stat_value = test_result.statistic
    p_value = test_result.pvalue

    ### if p-value <= 0.05, it means that there is a correlation between a numerical variable and predicted classes
    if p_value <= 0.05:
        test_results.append(abs(stat_value))


# %%

test_result_df = pd.DataFrame()
test_result_df['feature'] = num_cols
test_result_df['result'] = test_results
test_result_df = test_result_df.sort_values(by = 'result', ascending=False)

top_5_num_feature_cols = test_result_df['feature'].tolist()[:5]

# %%

all_selected_cols = top_5_cat_feature_cols + top_5_num_feature_cols + [label_str]

train_df = original_train_df[all_selected_cols]
test_df = test_df[all_selected_cols]


print('selected feature:')
print(top_5_cat_feature_cols + top_5_num_feature_cols)
print()

## for pre-processing

### only pre-process the columns that begin with 'AMT'
top_5_num_feature_cols = [s for s in top_5_num_feature_cols if s.startswith('AMT')]

### this feature is already in numerical format, so no need to do one-hot encoding
if 'NFLAG_INSURED_ON_APPROVAL' in top_5_cat_feature_cols:
    top_5_cat_feature_cols.remove('NFLAG_INSURED_ON_APPROVAL')

### this feature is already in numerical format, so no need to do one-hot encoding
if 'SELLERPLACE_AREA' in top_5_cat_feature_cols:
    top_5_cat_feature_cols.remove('SELLERPLACE_AREA')

print('pre-process the following features:')
print(top_5_cat_feature_cols + top_5_num_feature_cols)
print()

# %%

num_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)


col_transformer = ColumnTransformer(
    transformers=[
        ('num', num_transformer, top_5_num_feature_cols),
        ('cat', cat_transformer, top_5_cat_feature_cols)
    ],
    remainder= 'passthrough'
)

col_transformer.fit(train_df)

col_names = list(col_transformer.get_feature_names_out())

col_names = [s.replace('cat__','').replace('num__','').replace('remainder__','') for s in col_names]

    
train_data = col_transformer.transform(train_df).toarray()
test_data = col_transformer.transform(test_df).toarray()

train_df = pd.DataFrame(data=train_data, columns=col_names)
test_df = pd.DataFrame(data=test_data, columns=col_names)

# %%

train_df.to_csv(os.path.join(base_dir, task_name, 'train_processed_selected_features.csv'), index=False)
test_df.to_csv(os.path.join(base_dir, task_name, 'test_processed_selected_features.csv'), index=False)
