
import os, argparse, pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification, pipeline

from peft import PeftModel

from datasets import load_dataset

from torch.utils.data import DataLoader

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')
parser.add_argument('--ckpt_num', type = str, required=True)

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'bert': 'google-bert/bert-large-cased',
    'deberta': 'microsoft/deberta-v3-large'
}

args = parser.parse_args()

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')
ckpt_num = args.ckpt_num

output_model_dir = '../fine-tuned-model/{}-train-with-original-data'.format(model_name_arg)
real_model_path = os.path.join(output_model_dir, 'checkpoint-{}'.format(ckpt_num))

result_dir = '../generated_result/'
result_file_path = os.path.join(result_dir, 'result_from_{}.txt'.format(model_name_arg))

os.makedirs(result_dir, exist_ok=True)

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############


if model_name_arg == 'llama2-7b':
    test_batch_size = 1
else:
    test_batch_size = 8

idx2label = pickle.load(open('../dataset/cleaned/idx2class.pkl', 'rb'))
label2idx = pickle.load(open('../dataset/cleaned/class2idx.pkl', 'rb'))

dataset = load_dataset('csv', data_files={'test': os.path.join(data_dir,'test.csv')})

dataloader = DataLoader(dataset['test'], batch_size=test_batch_size)

############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


############## the below code is for generating predictions ##############

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

def preprocess_input_txt(input_text):
        
    input_text = tokenizer.decode(
                tokenizer.encode(
                    input_text,truncation=True, max_length = 900),
                skip_special_tokens=True)
    
    input_text = '<s>[INST]predict news category of the given news article below\n\n###news article\n\n {} \n\n[/INST]###news category: '.format(input_text)

    return input_text

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

    pipe = pipeline(task = "text-generation", model = model, tokenizer = tokenizer)

    for d in tqdm(dataloader):

        # print(d['text'])
        # print('-'*30)

        input_text = d['text'][0]

        input_text = preprocess_input_txt(input_text)
        
        output = pipe(input_text, max_new_tokens=5, return_full_text = False)

        output_text = output[0]['generated_text']
        output_text = output_text.replace('\n',' ').strip()

        print('input text:', input_text[:100],'...', input_text[-100:])
        print('output text:', output_text)
        print('-'*30)

        with open(result_file_path, 'a') as f:
            f.write(output_text+'\n')

else:
    print('loading encoder pre-trained model')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSequenceClassification.from_pretrained(
        real_model_path,
        # low_cpu_mem_usage=True,
        return_dict=True,
        # torch_dtype=torch.float16,
        # quantization_config=bnb_config,
        num_labels=13,
        id2label=idx2label,
        label2id=label2idx,
    )

    model.cuda()

    for batch in tqdm(dataloader):

        input_encodings = tokenizer.batch_encode_plus(batch['text'], pad_to_max_length=True, max_length=512)

        with torch.no_grad():
            logits = model(
                    input_ids = torch.tensor(input_encodings['input_ids'], dtype=torch.long).cuda(), 
                    attention_mask = torch.tensor(input_encodings['attention_mask'], dtype=torch.long).cuda()
                ).logits

        # print('logis:', logits)

        predicted_class_id = torch.argmax(logits, dim=1).tolist()

        print(predicted_class_id)

        predicted_class_id = list(predicted_class_id)
        predicted_class_id = [str(pred) for pred in predicted_class_id]

        with open(result_file_path, 'a') as f:
            f.write('\n'.join(predicted_class_id)+'\n')


# def preprocess_batch(batch):
#     if model_name_arg in ['llama2-7b', 'mistral-7b']:
#         batch['text'] = ['<s>[INST] {} [/INST] '.format(s) for s in batch['text']]
#     else:
#         batch['text'] = ['<s>{}</s>'.format(s) for s in batch['text']]

#     return batch


## Continue from here
## may need to have if...else for LLaMa-2 and encoder-only models

# for batch in tqdm(dataloader):

#     batch = preprocess_batch(batch)

#     input_encodings = tokenizer.batch_encode_plus(batch['text'], pad_to_max_length=True, max_length=512)
#     input_ids = input_encodings['input_ids'], 
#     attention_mask = input_encodings['attention_mask']

#     outs = model.generate(
#         input_ids=batch['input_ids'], 
#         attention_mask=batch['attention_mask'],
#         max_length=64,
#         early_stopping=True)
#     outs = [tokenizer.decode(ids) for ids in outs]

#     generated_title_list.extend(outs)