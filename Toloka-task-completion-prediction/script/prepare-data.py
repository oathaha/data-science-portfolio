#%%

import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#%%

data_dir = '../dataset/cleaned/'

df = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset.csv'))

#%%

## remove records that time-to-complete is marked as outlier

q1=df['completion-time-in-minutes'].quantile(0.25)

q3=df['completion-time-in-minutes'].quantile(0.75)

IQR=q3-q1

outlier_threshold = q3+1.5*IQR

print('outlier threshold:', outlier_threshold)

df = df[df['completion-time-in-minutes'] <= outlier_threshold]


#%%

## sort data by time
df['assignment_start_date'] = pd.to_datetime(df['assignment_start_date'])
df = df.sort_values(by='assignment_start_date')

# %%

## drop columns that are not used in model training

cols_to_drop = [
    'assignment_start_time',
    'assignment_status',
    'assignment_submit_time',
    'assignment_start_date', 'assignment_submit_date',
    'assignment_start_year', 'assignment_start_month',
    'assignment_start_day', 'assignment_start_hour',
    'assignment_start_minute', 'assignment_submit_year',
    'assignment_submit_month', 'assignment_submit_day',
    'assignment_submit_hour', 'assignment_submit_minute'
]

df = df.drop(cols_to_drop, axis=1)

# %%

train_end = round(len(df)*0.85)

train_df = df.iloc[0:train_end]
test_df = df.iloc[train_end:]

## save original one for later usage
train_df.to_csv(os.path.join(data_dir, 'train_original_df.csv'),index=False)
test_df.to_csv(os.path.join(data_dir, 'test_original_df.csv'),index=False)
df.to_csv(os.path.join(data_dir, 'cleaned_data_no_outlier_label.csv'),index=False)
#%%

## convert categorical features into numerical features

cat_cols = ['device_category', 'os_family', 'project_instruction_language', 'assignment_type']

num_cols = list(set(df.columns) - set(cat_cols) - set(['completion-time-in-minutes']))


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
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)

col_transformer.fit(train_df)

#%%

train_data = col_transformer.transform(train_df)

test_data = col_transformer.transform(test_df)

#%%

col_names = list(col_transformer.get_feature_names_out())

col_names = [s.replace('cat__','').replace('num__','') for s in col_names]

y_train = train_df['completion-time-in-minutes'].tolist()
y_test = test_df['completion-time-in-minutes'].tolist()


train_df = pd.DataFrame(train_data, columns=col_names)
train_df['completion-time-in-minutes'] = y_train

test_df = pd.DataFrame(test_data, columns=col_names)
test_df['completion-time-in-minutes'] = y_test

#%%

train_df.to_csv(os.path.join(data_dir, 'train_processed_df.csv'),index=False)
test_df.to_csv(os.path.join(data_dir, 'test_processed_df.csv'),index=False)

# %%
