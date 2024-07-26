#%%

import pandas as pd

from datasets import Dataset, load_dataset

#%%

df = pd.read_csv('../dataset/raw/193k.csv')

df.head()

# %%

print(df.loc[0]['title'])
print(df.loc[0]['text'])


# %%

df = df.sample(frac = 1.0, random_state=0)

train_end = round(len(df)*0.7)
valid_end = round(len(df)*0.85)

train_df = df.iloc[0:train_end]
valid_df = df.iloc[train_end: valid_end]
test_df = df.iloc[valid_end:]

assert len(df) == len(train_df) + len(valid_df) + len(test_df)

#%%

train_df.to_csv('../dataset/cleaned/train.csv', index = False)
valid_df.to_csv('../dataset/cleaned/valid.csv', index = False)
test_df.to_csv('../dataset/cleaned/test.csv', index = False)
