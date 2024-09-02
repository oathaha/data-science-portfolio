#%%

import pandas as pd

from datasets import Dataset, load_dataset

from multiprocessing import Pool

from nltk.tokenize import word_tokenize

import seaborn as sns

import matplotlib.pyplot as plt

#%%

df = pd.read_csv('../dataset/raw/193k.csv')

df.head()

#%%

def get_word_count(text):
    return len(word_tokenize(text))

articles = [[s] for s in df['text'].tolist()]
titles = [[s] for s in df['title'].tolist()]


pool = Pool(processes=16)
articles_word_count = pool.starmap(get_word_count, articles)
titles_word_count = pool.starmap(get_word_count, titles)


df['articles_word_count'] = articles_word_count
df['titles_word_count'] = titles_word_count

#%%

def get_article_word_count_range(wc):
    if wc <= 1000:
        return '<= 1,000'
    elif wc > 1000 and wc <= 2000:
        return '1,001 - 2,000'
    elif wc > 2000 and wc <= 3000:
        return '2,001 - 3,000'
    elif wc > 3000 and wc <= 4000:
        return '3,001 - 4,000'
    elif wc > 4000 and wc <= 5000:
        return '4,001 - 5,000'
    else:
        return '> 5,000'

def get_title_word_count_range(wc):
    if wc <= 5:
        return '<= 5'
    elif wc > 5 and wc <= 10:
        return '6 - 10'
    elif wc > 10 and wc <= 15:
        return '11 - 15'
    elif wc > 15 and wc <= 20:
        return '16 - 20'
    else:
        return '> 20'

#%%

df['articles_word_count_range'] = df['articles_word_count'].apply(lambda x: get_article_word_count_range(x))
df['titles_word_count_range'] = df['titles_word_count'].apply(lambda x: get_title_word_count_range(x))

#%%
plt.figure(figsize=(5,4))

plot = sns.histplot(data = df, x = 'articles_word_count', element="step")
plot.set(xlabel = 'Frequency', ylabel = 'Article word count')

plot

#%%
plt.figure(figsize=(5,4))

plot = sns.histplot(data = df, x = 'titles_word_count', element="step")
plot.set(xlabel = 'Frequency', ylabel = 'Title word count')

#%%

count_df = df.groupby('articles_word_count_range').size().reset_index()
count_df.columns = ['articles_word_count_range', 'count']
count_df = count_df.sort_values(by = 'count', ascending=False)

plt.figure(figsize=(5,4))

plot = sns.barplot(data = count_df, y = 'articles_word_count_range', x = 'count', hue = 'articles_word_count_range', order = ['<= 1,000', '1,001 - 2,000', '2,001 - 3,000', '3,001 - 4,000', '4,001 - 5,000', '> 5,000'])

plot.set(xlabel = 'Frequency', ylabel = 'Article word count')

for container in plot.containers:
    plot.bar_label(container)


plot


#%%

count_df = df.groupby('titles_word_count_range').size().reset_index()
count_df.columns = ['titles_word_count_range', 'count']
count_df = count_df.sort_values(by = 'count', ascending=False)

plt.figure(figsize=(5,4))

plot = sns.barplot(data = count_df, y = 'titles_word_count_range', x = 'count', hue = 'titles_word_count_range', order = ['<= 5', '6 - 10', '11 - 15', '16 - 20', '> 20'])

plot.set(xlabel = 'Frequency', ylabel = 'Title word count')

for container in plot.containers:
    plot.bar_label(container)

plot




#%%

df = df.drop(['articles_word_count, titles_word_count'], axis=1)


# %%

df = df.sample(frac = 1.0, random_state=0)

train_end = round(len(df)*0.7)
valid_end = round(len(df)*0.85)

train_df = df.iloc[0:train_end]
valid_df = df.iloc[train_end: valid_end]
valid_df_for_testing = df.sample(n=16)
test_df = df.iloc[valid_end:]

assert len(df) == len(train_df) + len(valid_df) + len(test_df)

#%%

train_df.to_csv('../dataset/cleaned/train.csv', index = False)
valid_df.to_csv('../dataset/cleaned/valid.csv', index = False)
valid_df_for_testing.to_csv('../dataset/cleaned/valid_for_testing.csv', index = False)
test_df.to_csv('../dataset/cleaned/test.csv', index = False)
