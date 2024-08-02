#%%

import pandas as pd
import os

#%%

data_dir = '../dataset/cleaned/'

df = pd.read_csv(os.path.join(data_dir, 'cleaned_dataset.csv'))

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

train_end = round(len(df)*0.7)

train_df = df.iloc[0:train_end]
test_df = df.iloc[train_end:]

# %%

train_df.to_csv(os.path.join(data_dir, 'train_df.csv'),index=False)
test_df.to_csv(os.path.join(data_dir, 'test_df.csv'),index=False)
# %%
