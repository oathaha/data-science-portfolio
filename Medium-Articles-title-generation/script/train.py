
#%%

## note: code for llm only, will add code for T5 later...

import os, argparse

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig

from peft import LoraConfig
from trl import SFTTrainer

from datasets import load_dataset


#%%

# dataset_name = "mlabonne/guanaco-llama2-1k"

# dataset = load_dataset(dataset_name)


#%%

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')

args = parser.parse_args()

#%%

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'mistral-7b': 'mistralai/Mistral-7B-v0.3'
}

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')

output_model_dir = '../fine-tuned-model/'

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid.csv')})


def preprocess_function(examples):
    
    inputs = examples['text']
    targets = examples['title']
    
    examples['text'] = '<s>[INST] {} [/INST] {} </s>'.format(inputs, targets)

    return examples


dataset = dataset.map(preprocess_function)


############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


############## load model for training ##############

compute_dtype = getattr(torch, "float16")


peft_config = LoraConfig(
    task_type="CAUSAL_LM", 
    inference_mode=False, 
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False
)

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config
)


train_batch_size = 16
eval_batch_size = 16
learning_rate = 2e-5

eval_every_step = round(0.1*len(dataset['train'])/train_batch_size)

training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=output_model_dir,
    evaluation_strategy = "steps",
    eval_steps = eval_every_step, ## evaluate every 10% of training dataset (in term of batch)
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    weight_decay=0.01,
    warmup_steps = 0.05*len(dataset['train']),
    save_strategy = 'steps',
    save_total_limit=5,
    save_steps = eval_every_step,
    group_by_length = True,
    metric_for_best_model = 'eval_loss', ## for early stopping
    load_best_model_at_end = True ## for early stopping
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold = 0.01)]
)

trainer.train()