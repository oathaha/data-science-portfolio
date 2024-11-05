from gensim.summarization.bm25 import BM25

from nltk import word_tokenize

import pandas as pd
import numpy as np

from tqdm import tqdm

from multiprocessing import Pool

## use nrows just for testing
train_df = pd.read_csv('../dataset/cleaned/train.csv')

test_df = pd.read_csv('../dataset/cleaned/test.csv')

row_nums = np.arange(0,len(train_df))

train_df['row_num'] = row_nums
train_input = train_df['text'].tolist()


print('building BM25 scorer')

train_corpus = [word_tokenize(s) for s in train_input]

bm25 = BM25(train_corpus)

print('building BM25 scorer done')

print('getting training samples for each test sample')

def get_example_for_test_sample(row):

    global train_df, row_nums

    test_input_tok_list = word_tokenize(row['text'])

    scores = bm25.get_scores(test_input_tok_list)

    score_df = pd.DataFrame()
    score_df['row_num'] = row_nums
    score_df['score'] = scores

    score_df = score_df.sort_values(by='score',ascending=False)
    score_df = score_df.head(3)

    top_3_row_nums = score_df['row_num'].tolist()

    subset_train_df = train_df[train_df['row_num'].isin(top_3_row_nums)]

    selected_train_input = subset_train_df['text'].tolist()
    seleted_train_output = subset_train_df['title'].tolist()

    data_row = {
        'test_input': row['text'],
        'test_title': row['title']
    }

    for k in range(0,len(selected_train_input)):
        data_row['sample_input_{}'.format(k+1)] = selected_train_input[k]
        data_row['sample_title_{}'.format(k+1)] = seleted_train_output[k]

    return data_row

test_data_rows = []

for idx, row in tqdm(test_df.iterrows()):
    test_data_rows.append([row])


pool = Pool(processes=8)
new_test_data_rows = pool.starmap(get_example_for_test_sample, test_data_rows)


new_test_df = pd.DataFrame(new_test_data_rows)

new_test_df.to_csv('../dataset/cleaned/test_for_prompting.csv', index=False)

