
## for bert score
import evaluate

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk import word_tokenize



import pandas as pd
import numpy as np

import os

# from transformers import BertTokenizer, BertForMaskedLM, BertModel
# from bert_score import BERTScorer

import argparse

### example of how to calculate corpus_bleu
# >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',
# ...         'ensures', 'that', 'the', 'military', 'always',
# ...         'obeys', 'the', 'commands', 'of', 'the', 'party']
# >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',
# ...          'ensures', 'that', 'the', 'military', 'will', 'forever',
# ...          'heed', 'Party', 'commands']
# >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',
# ...          'guarantees', 'the', 'military', 'forces', 'always',
# ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']
# >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',
# ...          'army', 'always', 'to', 'heed', 'the', 'directions',
# ...          'of', 'the', 'party']
# >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',
# ...         'interested', 'in', 'world', 'history']
# >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',
# ...          'because', 'he', 'read', 'the', 'book']

# >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]

# >>> hypotheses = [hyp1, hyp2]
# >>> corpus_bleu(list_of_references, hypotheses) 


### example of how to calculate meteor score
# hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
# reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
# round(single_meteor_score(reference1, hypothesis1),4)


#%%

parser = argparse.ArgumentParser()

parser.add_argument('--result_file_path', type = str, required=True)

args = parser.parse_args()

result_file_path = args.result_file_path
model_name = result_file_path.replace('../generated_result/result_from_','').replace('.txt','')

with open(result_file_path) as f:
    generated_title = f.readlines()

test_df = pd.read_csv('../dataset/cleaned/test.csv')

ref_title = test_df['title'].tolist()

#%%

tok_generated_title = [word_tokenize(s) for s in ref_title]
tok_ref_title = [[word_tokenize(s)] for s in ref_title]

eval_result_list = []


## compute BLEU score
scr = round(corpus_bleu(tok_ref_title, tok_generated_title),4)*100

eval_result_list.append({
    'model': model_name,
    'metric': 'BLEU-4',
    'value': scr
})

print('BLEU:', scr)

## compute METEOR score
scr_list = []

for ref, gen in zip(tok_ref_title, tok_generated_title):
    single_scr = single_meteor_score(ref[0], gen)
    scr_list.append(single_scr)

scr = round(np.mean(scr_list),4)*100

# scr = round(meteor_score(tok_ref_title, tok_generated_title),4)

eval_result_list.append({
    'model': model_name,
    'metric': 'METEOR',
    'value': scr
})

print('METEOR:', scr)


## compute ROUGE score
rouge_scr = evaluate.load('rouge')
results = rouge_scr.compute(predictions=generated_title, references=ref_title)
rouge_l = round(results['rougeL'],4)*100

eval_result_list.append({
    'model': model_name,
    'metric': 'ROUGE-L',
    'value': rouge_l
})

print(results)

## compute bertscore
bertscore = evaluate.load("bertscore")
results = bertscore.compute(predictions=generated_title, references=ref_title, lang="en")

prec = round(np.mean(results['precision']), 4)*100
rec = round(np.mean(results['recall']), 4)*100
f1 = round(np.mean(results['f1']), 4)*100

eval_result_list.append({
    'model': model_name,
    'metric': 'BERTScore-Precision',
    'value': prec
})

eval_result_list.append({
    'model': model_name,
    'metric': 'BERTScore-Recall',
    'value': rec
})

eval_result_list.append({
    'model': model_name,
    'metric': 'BERTScore-F1',
    'value': f1
})

print('BERT score')
print('Precision: {}, Recall: {}, F1: {}'.format(prec,rec,f1))


result_df = pd.DataFrame(eval_result_list)

save_dir = '../eval_result/'
os.makedirs(save_dir, exist_ok=True)

result_df.to_csv(os.path.join(save_dir, model_name+'.csv'), index=False)