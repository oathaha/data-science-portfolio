
#%%

import os 

import torch
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq, BitsAndBytesConfig

from peft import LoraConfig, TaskType
from trl import SFTTrainer

from datasets import load_dataset


#%%

dataset_name = "mlabonne/guanaco-llama2-1k"

dataset = load_dataset(dataset_name)


#%%


data_dir = '../dataset/cleaned/'
model_name = ''

dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid.csv')})


tokenizer = AutoTokenizer.from_pretrained(model_name)


compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_8bit=True
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.1
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=quant_config)

model = get_peft_model(model, peft_config)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)


def preprocess_function(examples):
    inputs = examples['title']
    targets = examples['text']

    ## for LLaMa-2 only, may need to change for other models
    model_inputs = '<s>[INST] {} [/INST] {}'.format(inputs, targets)

    # model_inputs = tokenizer(inputs, text_target=targets, max_length=1024, truncation=True)

    return model_inputs


dataset = dataset.map(preprocess_function, batch = True)


train_batch_size = 32
eval_batch_size = 32
learning_rate = 2e-5

training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir="model_checkpoint",
    eval_strategy="steps",
    learning_rate=learning_rate,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    weight_decay=0.01,
    warmup_steps = 0.05*len(dataset['train']),
    save_strategy = 'steps',
    save_total_limit=5,
    save_steps = 0.1*round(len(dataset['train'])/train_batch_size),
    # fp16=True,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()