#%%

import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from scipy.stats import pointbiserialr, spearmanr

import os

# %%

base_dir = '../dataset/cleaned/'

train_df = pd.read_csv(os.path.join(base_dir, 'train_original_df.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test_original_df.csv'))

original_train_df = train_df.copy()


#%%

cat_cols = ['device_category', 'os_family', 'project_instruction_language', 'assignment_type']

num_cols = list(set(train_df.columns) - set(cat_cols) - set(['completion-time-in-minutes']))

labels = train_df['completion-time-in-minutes']

#%%

for c in cat_cols:

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df[c])
    train_df[c] = label_encoder.transform(train_df[c])

#%%

def select_top_features(is_category_feature, col_list):
    test_results = []

    for c in col_list:
        if is_category_feature:
            test_result = pointbiserialr(labels, train_df[c])
        else:
            test_result = spearmanr(train_df[c], labels)

        stat_value = test_result.statistic
        p_value = test_result.pvalue

        if p_value <= 0.05:
            test_results.append({
                'feature': c,
                'result': abs(stat_value)
            })

    test_result_df = pd.DataFrame(test_results)

    test_result_df = test_result_df.sort_values(by = 'result', ascending=False)

    top_feature_cols = test_result_df['feature'].tolist()[:5]

    print(test_result_df)

    return top_feature_cols

# %%

top_5_cat_cols = select_top_features(True, cat_cols)

top_5_num_cols = select_top_features(False, num_cols)


# %%

all_selected_cols = top_5_cat_cols + top_5_num_cols + ['completion-time-in-minutes']

train_df = original_train_df[all_selected_cols]
test_df = test_df[all_selected_cols]


print('selected feature:')
print(top_5_cat_cols + top_5_num_cols)
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
        ('num', num_transformer, top_5_num_cols),
        ('cat', cat_transformer, top_5_cat_cols)
    ],
    remainder= 'passthrough'
)

col_transformer.fit(train_df)

col_names = list(col_transformer.get_feature_names_out())

col_names = [s.replace('cat__','').replace('num__','').replace('remainder__','') for s in col_names]

    
train_data = col_transformer.transform(train_df)
test_data = col_transformer.transform(test_df)

train_df = pd.DataFrame(data=train_data, columns=col_names)
test_df = pd.DataFrame(data=test_data, columns=col_names)

# %%

train_df.to_csv(os.path.join(base_dir, 'train_processed_selected_features.csv'), index=False)
test_df.to_csv(os.path.join(base_dir, 'test_processed_selected_features.csv'), index=False)

# %%
