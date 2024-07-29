
#%%

## note: code for llm only, will add code for T5 later...

import os, argparse

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from datasets import load_dataset


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

output_model_dir = '../fine-tuned-model/{}'.format(model_name)

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid.csv')})

## just for testing
# dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid_for_testing.csv')})

print(dataset)

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

peft_config = LoraConfig(
    task_type="CAUSAL_LM", 
    inference_mode=False, 
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    use_dora=True
)

compute_dtype = getattr(torch, "float16")


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

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

train_batch_size = 4
eval_batch_size = 8
learning_rate = 2e-5

## real one
eval_every_step = round(0.1*len(dataset['train'])/train_batch_size)


training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=output_model_dir,
    evaluation_strategy = "steps",
    eval_steps = eval_every_step, ## evaluate every 10% of training dataset (in term of batch)
    logging_strategy = 'steps',
    logging_first_step = True,
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    weight_decay=0.01,
    warmup_steps = 0.05*len(dataset['train']),
    save_strategy = 'steps',
    save_total_limit=7,
    save_steps = eval_every_step,
    group_by_length = True,
    fp16=True,
    metric_for_best_model = 'eval_loss', ## for early stopping
    load_best_model_at_end = True ## for early stopping
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    # peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold = 0.01)]
)

trainer.train()