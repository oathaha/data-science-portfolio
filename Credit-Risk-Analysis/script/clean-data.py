#%%

import pandas as pd

import os

#%%

df = pd.read_csv('../dataset/raw/previous_application.csv')

save_dir = '../dataset/cleaned/'

os.makedirs(save_dir, exist_ok=True)

#%%

## drop duplicate
df = df.drop_duplicates()

# %%

## check nan value
print(df.isna().sum(axis=0))

# %%

## replace nan value with suitable values

df['AMT_ANNUITY'] = df['AMT_ANNUITY'].fillna(0)
df['AMT_CREDIT'] = df['AMT_CREDIT'].fillna(0)
df['AMT_DOWN_PAYMENT'] = df['AMT_DOWN_PAYMENT'].fillna(0)
df['AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'].fillna(0)

## assume that no interest rate for loan
df['RATE_DOWN_PAYMENT'] = df['RATE_DOWN_PAYMENT'].fillna(0)
df['RATE_INTEREST_PRIMARY'] = df['RATE_INTEREST_PRIMARY'].fillna(0)
df['RATE_INTEREST_PRIVILEGED'] = df['RATE_INTEREST_PRIVILEGED'].fillna(0)

df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('XNA')
df['CNT_PAYMENT'] = df['CNT_PAYMENT'].fillna(0) # assume that 0 means loan can be paid anytime
df['PRODUCT_COMBINATION'] = df['PRODUCT_COMBINATION'].fillna('XNA')
df['NFLAG_INSURED_ON_APPROVAL'] = df['NFLAG_INSURED_ON_APPROVAL'].fillna(0) # assume that the missing values are 0 

# %%

## note: some columns still have nan value but they will be dropped anyway, so no need to clean
print(df.isna().sum())

# %%

## save cleaned dataset
df.to_csv(os.path.join(save_dir, 'cleaned_data.csv'), index=False)
