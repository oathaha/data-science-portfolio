import os, argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from peft import PeftModel

from datasets import load_dataset


parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')

args = parser.parse_args()

#%%

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b',
    'mistral-7b': 'mistralai/Mistral-7B-v0.3'
}

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')
new_model_name = '' ## will add later...

output_model_dir = '../fine-tuned-model/'
real_model_path = os.path.join(output_model_dir, new_model_name)

result_dir = '../generated_result/'
result_file_path = os.path.join(result_dir, 'result_from_{}.txt'.format(model_name))

os.makedirs(result_dir, exist_ok=True)



if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

dataset = load_dataset('csv', data_files={'test': os.path.join(data_dir,'test.csv')})


############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


############## load model for training ##############
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, real_model_path)
model = model.merge_and_unload()

############## generate results from model ##############

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=64)

generated_title_list = []

for d in dataset:
    input_txt = d['text']

    result = pipe("<s>[INST] {} [/INST]".format(input_txt))
    generated_title = result[0]['generated_text']

    generated_title_list.append(generated_title)


with open(result_file_path, 'w') as f:
    f.write('\n'.join(generated_title_list))