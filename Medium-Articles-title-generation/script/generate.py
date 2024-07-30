import os, argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

from peft import PeftModel

from datasets import load_dataset

from torch.utils.data import DataLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')
parser.add_argument('--ckpt_dir', type = str, required=True)

args = parser.parse_args()

#%%

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.3',
    't5': 'google-t5/t5-3b',
    'bart': 'facebook/bart-large'
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

max_output_len = 64

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

dataset = load_dataset('csv', data_files={'test': os.path.join(data_dir,'test.csv')})


dataloader = DataLoader(dataset, batch_size=4)

############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

if model_name_arg in ['llama2-7b', 'mistral-7b']:
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
    print('loading encoder-decoder pre-trained model')

    model = AutoModelForSeq2SeqLM.from_pretrained(
        real_model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

generated_title_list = []



## continue from here...
for batch in tqdm(dataloader):
    input_encodings = tokenizer.batch_encode_plus(batch, pad_to_max_length=True, max_length=512)
    input_ids = input_encodings['input_ids'], 
    attention_mask = input_encodings['attention_mask']

    outs = model.generate(
        input_ids=batch['input_ids'], 
        attention_mask=batch['attention_mask'],
        max_length=64,
        early_stopping=True)
    outs = [tokenizer.decode(ids) for ids in outs]

    generated_title_list.extend(outs)

# # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=64)


# for d in dataset:
#     input_txt = d['text']

#     result = pipe("<s>[INST] {} [/INST]".format(input_txt))
#     generated_title = result[0]['generated_text']

#     generated_title_list.append(generated_title)


#%%


with open(result_file_path, 'w') as f:
    f.write('\n'.join(generated_title_list))