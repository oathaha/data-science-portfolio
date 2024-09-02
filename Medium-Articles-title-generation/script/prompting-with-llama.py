
import os, argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from datasets import load_dataset

from torch.utils.data import DataLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()

## zero-shot or few-shot only
parser.add_argument('--prompting_technique', type = str, required=True)

args = parser.parse_args()

prompting_technique = args.prompting_technique

data_dir = '../dataset/cleaned/'
model_name_arg = 'llama2-7b'
model_name = 'meta-llama/Llama-2-7b-hf'


result_dir = '../generated_result/'
result_file_path = os.path.join(result_dir, 'result_from_{}_{}.txt'.format(model_name_arg, prompting_technique))
csv_output_file_path = os.path.join(result_dir, 'result_from_{}_{}.csv'.format(model_name_arg, prompting_technique))

os.makedirs(result_dir, exist_ok=True)



if prompting_technique not in ['zero-shot', 'few-shot']:
    print('wrong prompting technique')
    print('prompting technique must be either zero-shot or few-shot')
    exit(0)

max_input_lens = {
    'zero-shot': 3950,
    'few-shot': 980
}

max_input_len = max_input_lens[prompting_technique]

if prompting_technique == 'zero-shot':
    test_df = pd.read_csv('../dataset/cleaned/test.csv')
else:
    test_df = pd.read_csv('../dataset/cleaned/test_for_prompting.csv')

############## load dataset ##############



############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


############## the below code is for generating predictions ##############

prompt_template_for_example = '''[INST]Below is the article

{}

Generate title for this article[/INST]

Title: {}\n\n'''

prompt_template_for_real_test_sample = '''[INST]Below is the article

{}

Generate title for this article[/INST]

Title: '''

def truncate_input(input_text):
    input_text = tokenizer.decode(
        tokenizer.encode(
            input_text,truncation=True, max_length = max_input_len),
        skip_special_tokens=True)

    return input_text

## input_row: row from a dataframe
def create_prompt(input_row):

    if prompting_technique == 'zero-shot':
        input_text = truncate_input(input_row['text'])
        prompt = prompt_template_for_real_test_sample.format(input_text)

    else:
        input_text = truncate_input(input_row['test_input'])
        prompt = ''

        for i in range(0,3):
            input_sample = input_row['sample_input_{}'.format(i+1)]
            label_sample = input_row['sample_title_{}'.format(i+1)]

            input_sample = truncate_input(input_sample)

            sample_prompt = prompt_template_for_example.format(input_sample, label_sample)

            prompt = prompt + sample_prompt

        prompt = prompt + prompt_template_for_real_test_sample.format(input_text)

    return prompt


print('loading LLM')

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)

pipe = pipeline(task = "text-generation", model = model, tokenizer = tokenizer)


data_rows = []

for idx, row in tqdm(test_df.iterrows()):

    input_prompt = create_prompt(row)

    output = pipe(input_prompt, max_new_tokens=3, return_full_text = False)

    output_text = output[0]['generated_text']
    output_text = output_text.replace('\n',' ').strip()

    data_row = {
        'prompt': input_prompt,
        'output': output_text
    }

    data_rows.append(data_row)

    with open(result_file_path, 'a') as f:
        f.write(output_text+'\n')

    df = pd.DataFrame(data_rows)
    df.to_csv(csv_output_file_path, index=False)

# for d in tqdm(dataloader):

#     input_text = d['text'][0]

#     input_text = preprocess_input_txt(input_text)
    
#     output = pipe(input_text, max_new_tokens=5, return_full_text = False)

#     output_text = output[0]['generated_text']
#     output_text = output_text.replace('\n',' ').strip()

#     print('input text:', input_text[:100],'...', input_text[-100:])
#     print('output text:', output_text)
#     print('-'*30)

#     with open(result_file_path, 'a') as f:
#         f.write(output_text+'\n')
