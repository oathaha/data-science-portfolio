import os, argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig, pipeline

from peft import PeftModel, AutoPeftModelForCausalLM

from datasets import load_dataset

from torch.utils.data import DataLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')
parser.add_argument('--ckpt_num', type = str, required=True)

args = parser.parse_args()

#%%

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    't5': 'google-t5/t5-3b',
    'bart': 'facebook/bart-large',
    'long-t5': 'google/long-t5-tglobal-xl'
}

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')
ckpt_num = args.ckpt_num

output_model_dir = '../fine-tuned-model/{}'.format(model_name_arg)
real_model_path = os.path.join(output_model_dir, 'checkpoint-{}'.format(ckpt_num))

result_dir = '../generated_result/'
result_file_path = os.path.join(result_dir, 'result_from_{}.txt'.format(model_name_arg))

os.makedirs(result_dir, exist_ok=True)

max_output_len = 50

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

if model_name_arg in 'llama2-7b':
    test_batch_size = 1
else:
    test_batch_size = 8

dataset = load_dataset('csv', data_files={'test': os.path.join(data_dir,'test.csv')})

dataloader = DataLoader(dataset['test'], batch_size=test_batch_size, shuffle=False)

############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


############## load model and its checkpoint ##############

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

if model_name_arg == 'llama2-7b':
    print('loading LLM')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

    model = PeftModel.from_pretrained(model, real_model_path)
    model = model.merge_and_unload()

else:
    print('loading encoder-decoder pre-trained model')

    if model_name_arg == 'long-t5':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )

        print('load LORA for model')
        
        model = PeftModel.from_pretrained(model, real_model_path)
        model = model.merge_and_unload()

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        real_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )


def preprocess_batch(batch):
    
    batch['text'] = ['<s>{}</s>'.format(s) for s in batch['text']]

    return batch


def preprocess_input_txt(input_txt):
    input_txt = tokenizer.decode(
                    tokenizer.encode(
                        input_txt,truncation=True, max_length = 3950),
                    skip_special_tokens=True)

    input_txt = '<s>[INST]generate title from the given article below\n\n###article\n\n {} \n\n[/INST]###title: '.format(input_txt)

    return input_txt

if model_name_arg == 'llama2-7b':

    pipe = pipeline(task = "text-generation", model = model, tokenizer = tokenizer)

    for d in tqdm(dataloader):

        input_text = d['text'][0]

        input_text = preprocess_input_txt(input_text)
        
        output = pipe(input_text, max_new_tokens=max_output_len, return_full_text = False)

        output_text = output[0]['generated_text']
        output_text = output_text.replace('\n',' ').strip()

        with open(result_file_path, 'a') as f:
            f.write(output_text+'\n')
            

else:

    for batch in tqdm(dataloader):

        batch = preprocess_batch(batch)

        input_encodings = tokenizer.batch_encode_plus(batch['text'], pad_to_max_length=True, max_length=512)

        outs = model.generate(
            input_ids=torch.tensor(input_encodings['input_ids'],dtype=torch.long).cuda(), 
            attention_mask=torch.tensor(input_encodings['attention_mask'],dtype=torch.long).cuda(),
            max_length=max_output_len,
            early_stopping=False)

        outs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]


        with open(result_file_path, 'a') as f:
            f.write('\n'.join(outs)+'\n')
