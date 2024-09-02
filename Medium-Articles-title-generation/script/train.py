
#%%
import os, argparse

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig, Trainer, AutoModelForSeq2SeqLM

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
    't5': 'google-t5/t5-3b',
    'bart': 'facebook/bart-large',
    'long-t5': 'google/long-t5-tglobal-xl'
}

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')

output_model_dir = '../fine-tuned-model/{}'.format(model_name_arg)

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid.csv')})

## just for testing
# dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid_for_testing.csv')})


print(dataset)


############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token



############## load model for training ##############

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)


train_batch_size = 8
eval_batch_size = 8
learning_rate = 1e-5

## real one
# eval_every_step = round(0.1*len(dataset['train'])/train_batch_size)
eval_every_step = 3050 ## total steps are 30516 as seen from screen.


if model_name_arg == 'llama2-7b':
    fp16 = True

else:
    fp16 = False

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
    fp16=fp16,
    metric_for_best_model = 'eval_loss', ## for early stopping
    load_best_model_at_end = True ## for early stopping
)




#%%

def train_LLM():

    global dataset

    def preprocess_function(examples):
        
        inputs = examples['text']
        targets = examples['title']

        ## truncate input and target 

        inputs = tokenizer.decode(
                    tokenizer.encode(
                        inputs,truncation=True, max_length = 800),
                    skip_special_tokens=True)
        targets = tokenizer.decode(
                    tokenizer.encode(
                        targets,truncation=True, max_length = 128),
                    skip_special_tokens=True)
        
        examples['text'] = '<s>[INST]generate title from the given article below\n\n###article\n\n {} \n\n[/INST]###title: {} </s>'.format(inputs, targets)

        return examples

    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    dataset = dataset.map(preprocess_function)

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

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)

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


def train_enc_dec_model():

    global dataset

    if 't5' in model_name_arg:
        max_input_len = 512
    else:
        max_input_len = 800

    # tokenize the examples
    def convert_to_features(example_batch):
        model_inputs = tokenizer(example_batch['text'], max_length=max_input_len, truncation=True, pad_to_max_length = True)

        labels = tokenizer(text_target=example_batch['title'], max_length=128, truncation=True, pad_to_max_length = True)

        model_inputs["labels"] = labels["input_ids"]

        # inputs = example_batch['text']
        # targets = example_batch['title']
        # model_inputs = tokenizer(
        #     inputs, text_target=targets, max_length=512, truncation=True, pad_to_max_length=True
        # )

        return model_inputs

    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    dataset = dataset.map(convert_to_features, batched=True)

    peft_config = LoraConfig(
        task_type="SEQ_2_SEQ_LM", 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
        bias="none",
        use_dora=True,
        target_modules=["q","k","v","o"],
    )

    if model_name_arg == 'long-t5':

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            quantization_config=bnb_config
        )

        # print(model)
        # exit()

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)

        print('enable inputs require grad')
        model.base_model.model.encoder.enable_input_require_grads()
        model.base_model.model.decoder.enable_input_require_grads()

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        args=training_args,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold = 0.01)]
    )

    trainer.train()


#%%

if model_name_arg in ['llama2-7b', 'mistral-7b']:
    print('training LLM')
    train_LLM()

else:
    print('training encoder-decoder pre-trained model')
    train_enc_dec_model()