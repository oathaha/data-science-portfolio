import os, argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig, pipeline

from peft import PeftModel, AutoPeftModelForCausalLM

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




############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

if model_name_arg == 'llama2-7b':
    print('loading LLM')
    # model = AutoPeftModelForCausalLM.from_pretrained(real_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

    # model.load_adapter(real_model_path)

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


def preprocess_batch(batch):
    if model_name_arg == 'llama2-7b':

        # inputs = tokenizer.decode(
        #             tokenizer.encode(
        #                 inputs,truncation=True, max_length = 970),
        #             skip_special_tokens=True)

        batch['text'] = [tokenizer.decode(
                    tokenizer.encode(
                        s,truncation=True, max_length = 970),
                    skip_special_tokens=True) for s in batch['text']]

        batch['text'] = ['<s>[INST] {} [/INST] '.format(s) for s in batch['text']]

    else:
        batch['text'] = ['<s>{}</s>'.format(s) for s in batch['text']]

    return batch


if model_name_arg == 'llama2-7b':
    dataloader = DataLoader(dataset['test'], batch_size=1)

    pipe = pipeline(task='summarization',model=model, tokenizer=tokenizer, framework='pt', device_map='auto')

    for d in tqdm(dataloader):

        # print(d['text'])

        input_text = d['text'][0]
        input_text = tokenizer.decode(
                        tokenizer.encode(
                            input_text,truncation=True, max_length = 4090),
                        skip_special_tokens=True)

        output = pipe(input_text)

        print(output)

        generated_title_list.append(output)

        # messages = [
        #     {"role": "user", 
        #     "content": },
        # ]

        # # prepare the messages for the model
        # input_ids = tokenizer.apply_chat_template(messages, truncation=True, return_tensors="pt").to("cuda")

        # outputs = model.generate(
        #             input_ids=input_ids,
        #             max_new_tokens=max_output_len,
        #             do_sample=True,
        #             temperature=0.7,
        #             top_k=10,
        #             top_p=1.0
        #         )

        # out_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # generated_title_list.append(output_text)

        break


else:
    for batch in tqdm(dataloader):

        batch = preprocess_batch(batch)

        input_encodings = tokenizer.batch_encode_plus(batch['text'], pad_to_max_length=True, max_length=512)
        # input_ids = input_encodings['input_ids'], 
        # attention_mask = input_encodings['attention_mask']

        outs = model.generate(
            input_ids=torch.tensor(input_encodings['input_ids'],dtype=torch.long), 
            attention_mask=torch.tensor(input_encodings['attention_mask'],dtype=torch.long),
            max_length=512,
            early_stopping=False)
        outs = [tokenizer.decode(ids) for ids in outs]

        generated_title_list.extend(outs)

        break

# # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=64)


# for d in dataset:
#     input_txt = d['text']

#     result = pipe("<s>[INST] {} [/INST]".format(input_txt))
#     generated_title = result[0]['generated_text']

#     generated_title_list.append(generated_title)


#%%


with open(result_file_path, 'w') as f:
    f.write('\n'.join(generated_title_list))