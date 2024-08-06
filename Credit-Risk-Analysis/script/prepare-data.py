#%%

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#%%

save_dir = '../dataset/cleaned/'

## predict whether loan is approved or not
save_dir_app_ref = os.path.join(save_dir,'loan_approval_prediction')

## predict whether loan should be reviewed now or later
save_dir_review_priority = os.path.join(save_dir,'review_priority_prediction')

os.makedirs(save_dir_app_ref, exist_ok=True)
os.makedirs(save_dir_review_priority, exist_ok=True)

df = pd.read_csv(os.path.join(save_dir, 'cleaned_data.csv'))

# %%


## remove unused columns
## these columns are not related to decision making about whether to grant a loan in reality
cols_to_drop = [
    'SK_ID_PREV',
    'SK_ID_CURR',
    'DAYS_FIRST_DRAWING',
    'DAYS_FIRST_DUE',
    'DAYS_LAST_DUE_1ST_VERSION',
    'DAYS_LAST_DUE',
    'DAYS_TERMINATION',
    'DAYS_DECISION',
    'WEEKDAY_APPR_PROCESS_START',
    'HOUR_APPR_PROCESS_START',
    'CODE_REJECT_REASON',
    'FLAG_LAST_APPL_PER_CONTRACT',
    'NFLAG_LAST_APPL_IN_DAY',
    'NAME_PRODUCT_TYPE',
    'CHANNEL_TYPE',
    'NAME_PAYMENT_TYPE',
    'AMT_CREDIT'
]

df = df.drop(cols_to_drop, axis=1)

df = df.drop_duplicates()

#%%

## shuffle dataset
df = df.sample(frac = 1.0)

#%%

## prepare pipeline for data preprocessing

num_cols = [
    'AMT_ANNUITY',
    'AMT_APPLICATION',
    'AMT_DOWN_PAYMENT',
    'AMT_GOODS_PRICE',
]

cat_cols = [
    'NAME_CONTRACT_TYPE',
    'NAME_TYPE_SUITE',
    'NAME_CLIENT_TYPE',
    'NAME_GOODS_CATEGORY',
    'NAME_PORTFOLIO',
    'NAME_CASH_LOAN_PURPOSE',
    'NAME_SELLER_INDUSTRY',
    'NAME_YIELD_GROUP',
    'PRODUCT_COMBINATION',
]

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

def transform_data(train_df, test_df):

    train_idx = np.arange(0,len(train_df))
    test_idx = np.arange(0,len(test_df))
    
    col_transformer = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder= 'passthrough'
    )

    col_transformer.fit(train_df)

    col_names = list(col_transformer.get_feature_names_out())

    col_names = [s.replace('cat__','').replace('num__','').replace('remainder__','') for s in col_names]

    # print(col_names)
        
    train_data = col_transformer.transform(train_df).toarray()
    test_data = col_transformer.transform(test_df).toarray()
    
    ## shape: n_row x n_cols
    # print(type(train_data), train_data.shape)
    # print(test_data.shape)

    train_df = pd.DataFrame(data=train_data, columns=col_names)
    test_df = pd.DataFrame(data=test_data, columns=col_names)

    return train_df, test_df

#%%

df_app_ref = df[df['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])]

train_end = round(len(df_app_ref)*0.85)

df_app_ref['is_approved'] = df_app_ref['NAME_CONTRACT_STATUS'] == 'Approved'

df_app_ref = df_app_ref.drop('NAME_CONTRACT_STATUS', axis=1)

train_df = df_app_ref.iloc[0:train_end]
test_df = df_app_ref.iloc[train_end:]

df_app_ref.to_csv(os.path.join(save_dir_app_ref,'processed_data.csv'), index=False)
train_df.to_csv(os.path.join(save_dir_app_ref,'train_original_data.csv'), index=False)
test_df.to_csv(os.path.join(save_dir_app_ref,'test_original_data.csv'), index=False)

train_df, test_df = transform_data(train_df, test_df)

train_df.to_csv(os.path.join(save_dir_app_ref,'train_processed_data.csv'), index=False)
test_df.to_csv(os.path.join(save_dir_app_ref,'test_processed_data.csv'), index=False)

# %%

# df_app_ref = df[df['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])]

df['is_high_priority'] = df['NAME_CONTRACT_STATUS'].isin(['Approved', 'Refused'])

df = df.drop('NAME_CONTRACT_STATUS', axis=1)

#%%

train_end = round(len(df)*0.85)

train_df = df.iloc[0:train_end]
test_df = df.iloc[train_end:]

df.to_csv(os.path.join(save_dir_review_priority,'processed_data.csv'), index=False)
train_df.to_csv(os.path.join(save_dir_review_priority,'train_original_data.csv'), index=False)
test_df.to_csv(os.path.join(save_dir_review_priority,'test_original_data.csv'), index=False)

train_df, test_df = transform_data(train_df, test_df)

train_df.to_csv(os.path.join(save_dir_review_priority,'train_processed_data.csv'), index=False)
test_df.to_csv(os.path.join(save_dir_review_priority,'test_processed_data.csv'), index=False)
# %%
