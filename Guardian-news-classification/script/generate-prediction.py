
import os, argparse, pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification

from peft import PeftModel

from datasets import load_dataset

from torch.utils.data import DataLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')
parser.add_argument('--ckpt_dir', type = str, required=True)

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'bert': 'google-bert/bert-large-cased',
    'deberta': 'microsoft/deberta-v3-large'
}

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')
ckpt_dir = args.ckpt_dir

output_model_dir = '../fine-tuned-model/{}'.format(model_name_arg)
real_model_path = os.path.join(output_model_dir, ckpt_dir)

result_dir = '../generated_result/'
result_file_path = os.path.join(result_dir, 'result_from_{}.txt'.format(model_name_arg))

os.makedirs(result_dir, exist_ok=True)

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

idx2label = pickle.load(open('../dataset/cleaned/idx2class.pkl', 'rb'))
label2idx = pickle.load(open('../dataset/cleaned/class2idx.pkl', 'rb'))

dataset = load_dataset('csv', data_files={'test': os.path.join(data_dir,'test.csv')})

dataloader = DataLoader(dataset['test'], batch_size=4)

############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

if model_name_arg == 'llama2-7b':
    print('loading LLM')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, real_model_path)
    model = model.merge_and_unload()

else:
    print('loading encoder pre-trained model')

    model = AutoModelForSequenceClassification.from_pretrained(
        real_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        num_labels=13,
        id2label=idx2label,
        label2id=label2idx,
    )


def preprocess_batch(batch):
    if model_name_arg in ['llama2-7b', 'mistral-7b']:
        batch['text'] = ['<s>[INST] {} [/INST] '.format(s) for s in batch['text']]
    else:
        batch['text'] = ['<s>{}</s>'.format(s) for s in batch['text']]

    return batch


## Continue from here
## may need to have if...else for LLaMa-2 and encoder-only models

for batch in tqdm(dataloader):

    batch = preprocess_batch(batch)

    input_encodings = tokenizer.batch_encode_plus(batch['text'], pad_to_max_length=True, max_length=512)
    input_ids = input_encodings['input_ids'], 
    attention_mask = input_encodings['attention_mask']

    outs = model.generate(
        input_ids=batch['input_ids'], 
        attention_mask=batch['attention_mask'],
        max_length=64,
        early_stopping=True)
    outs = [tokenizer.decode(ids) for ids in outs]

    generated_title_list.extend(outs)