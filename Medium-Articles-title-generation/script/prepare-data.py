#%%

import pandas as pd
# from datasets import Dataset

#%%

df = pd.read_csv('../dataset/raw/193k.csv')

df.head()

# %%

print(df.loc[0]['title'])
print(df.loc[0]['text'])


# %%
