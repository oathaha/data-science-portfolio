#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

#%%

base_dir = '../eval_result/'
result_files = os.listdir(base_dir)

# %%

all_df = []

for d in result_files:
    df = pd.read_csv(os.path.join(base_dir, d))

    all_df.append(df)
# %%

result_df = pd.concat(all_df)
result_df = result_df[~result_df['metric'].isin(['BERTScore-Precision', 'BERTScore-Recall'])]

result_df = result_df.sort_values(by = ['metric', 'value'], ascending=False)
# %%

new_labels = [
    'T5', 'BART', 'Long-T5',
    'LLaMa-2-7B (Few-shot)',
    'LLaMa-2-7B (Fine-tuned)',
    'LLaMa-2-7B (Zero-shot)'
]

g = sns.barplot(data = result_df, x='metric', y='value', hue='model')

for t, l in zip(g.legend_.texts, new_labels):
    t.set_text(l)

sns.move_legend(
    g, "lower center",
    bbox_to_anchor=(0.5, -.4),
    ncol=3,
    title='Model', frameon=False,
)

plt.xlabel('')