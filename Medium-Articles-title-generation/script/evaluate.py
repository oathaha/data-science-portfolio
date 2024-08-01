
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize

## for bert score
from evaluate import load

import numpy as np

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

with open(result_file_path) as f:
    generated_title = f.readlines()

test_df = pd.read_csv('../dataset/cleaned/test.csv')

ref_title = test_df['title'].tolist()

#%%

tok_generated_title = [word_tokenize(s) for s in ref_title]
tok_ref_title = [[word_tokenize(s)] for s in ref_title]


## compute BLEU score
scr = scr = round(corpus_bleu(tok_ref_title, tok_generated_title),4)

print('BLEU:', scr)

## compute METEOR score
scr = round(meteor_score(tok_ref_title, tok_generated_title),4)

print('METEOR:', scr)

## compute bertscore
bertscore = load("bertscore")
results = bertscore.compute(predictions=generated_title, references=ref_title, lang="en")

prec = round(np.mean(results['precision']), 4)
rec = round(np.mean(results['recall']), 4)
f1 = round(np.mean(results['f1']), 4)

print('BERT score')
print('Precision: {}, Recall: {}, F1: {}'.format(prec,rec,f1))