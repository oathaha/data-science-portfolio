#%%

import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef

import pickle, argparse

# %%

gt = pd.read_csv('../dataset/cleaned/test.csv')

with open('../dataset/cleaned/class2idx.pkl', 'rb') as f:
    class2idx = pickle.load(f)

idx_list = list(class2idx.values())
idx_str_list = [str(s) for s in idx_list]

# %%

parser = argparse.ArgumentParser()

parser.add_argument('--result_file_path', type = str, required=True)

args = parser.parse_args()

result_file_path = args.result_file_path
model_name = result_file_path.replace('../generated_result/result_from_','').replace('.txt','')

#%%

print('evaluating results from', model_name)

true_labels_int = gt['label'].tolist()

with open(result_file_path) as f:
    predictions = f.readlines()

if model_name == 'llama2-7b':
    ## some lines contain more than 1 words, so get only the first word

    predictions = [s.strip() if len(s.split()) > 0 else '[BLANK]' for s in predictions]

    predictions = [s.split()[0].strip() for s in predictions]

    ## some lines contain uppercase letters, so make them lowercase so that they match keys in dictionary
    predictions = [s.lower() for s in predictions]

    predictions = [class2idx.get(s, -1) for s in predictions]

    labels = idx_list + [-1]

    print(predictions[:50])

elif 'shot' in model_name:
    
    predictions = [s.strip() if len(s.split()) > 0 else '[BLANK]' for s in predictions]

    predictions = [s.split()[0].replace(',','').replace('.','').strip() for s in predictions]
    
    predictions = [int(s)-1 if s in idx_str_list else -1 for s in predictions]

    labels = idx_list + [-1]

    print(predictions[:50])

else:
    predictions = [int(s.strip()) for s in predictions]

    labels = None


print(classification_report(
    true_labels_int, 
    predictions, 
    labels = labels,
    target_names=list(class2idx.keys())
    ))

print('MCC:', round(matthews_corrcoef(true_labels_int, predictions),4)*100)