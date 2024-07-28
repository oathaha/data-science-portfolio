#%%

import pandas as pd
import seaborn as sns

import pickle

#%%

df = pd.read_parquet('../dataset/raw/all_Article_df.parquet')

# %%

df.head()

# %%

class_count = df.groupby('label').size().reset_index()
class_count.columns = ['label', 'count']
class_count = class_count.sort_values(by = 'count', ascending=False)

plot = sns.barplot(data = class_count, x = 'label', y = 'count')

plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
plot.bar_label(plot.containers[0])

plot

#%%

## convert class name to label

class_list = class_count['label'].tolist()

class2idx = {}
idx2class = {}

for i in range(0,len(class_list)):
    class2idx[class_list[i]] = i
    idx2class[i] = class_list[i]


#%%

df['label_int'] = df['label'].apply(lambda x: class2idx[x])
df = df.drop('url', axis=1)
df = df.dropna()


# %%

## since dataset is imbalance, I use stratified sampling to ensure that all classes are in training and validation set

train_df_list, valid_df_list, test_df_list = [], [], []

for class_name, sub_df in df.groupby('label'):
    train_end = round(len(sub_df)*0.7)
    valid_end = round(len(sub_df)*0.85)

    train_df = sub_df.iloc[0:train_end]
    valid_df = sub_df.iloc[train_end: valid_end]
    test_df = sub_df.iloc[valid_end:]

    train_df_list.append(train_df)
    valid_df_list.append(valid_df)
    test_df_list.append(test_df)

#%%

final_train_df = pd.concat(train_df_list).sample(frac = 1.0, random_state=0)
final_valid_df = pd.concat(valid_df_list).sample(frac = 1.0, random_state=0)
final_test_df = pd.concat(test_df_list).sample(frac = 1.0, random_state=0)

#%%

train_df.to_csv('../dataset/cleaned/train.csv', index = False)
valid_df.to_csv('../dataset/cleaned/valid.csv', index = False)
test_df.to_csv('../dataset/cleaned/test.csv', index = False)

# %%