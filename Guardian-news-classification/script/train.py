
#%%
import os, argparse, pickle

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, EarlyStoppingCallback, BitsAndBytesConfig, Trainer, AutoModelForSequenceClassification

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from datasets import load_dataset


#%%

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type = str, default = 'llama2-7b')
parser.add_argument('--handle_imb_data', type = bool, default = False)

args = parser.parse_args()

#%%

model_names = {
    'llama2-7b': 'meta-llama/Llama-2-7b-hf',
    'bert': 'google-bert/bert-large-cased',
    'deberta': 'microsoft/deberta-v3-large'
}

data_dir = '../dataset/cleaned/'
model_name_arg = args.model_name
model_name = model_names.get(model_name_arg, '')

output_model_dir = '../fine-tuned-model/{}-{}'.format(model_name_arg)

if model_name == '':
    print('wrong model name.')
    print('the model names must be in the list:', list(model_names.keys()))
    exit(0)

############## load dataset ##############

dataset = load_dataset('csv', data_files={'train': os.path.join(data_dir,'train.csv'), 'valid': os.path.join(data_dir,'valid.csv')})

idx2label = pickle.load(open('../dataset/cleaned/idx2class.pkl', 'rb'))
label2idx = pickle.load(open('../dataset/cleaned/class2idx.pkl', 'rb'))

print(dataset)


############## load tokenizer ##############

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


############## load model for training ##############

train_batch_size = 8
eval_batch_size = 8
learning_rate = 1e-5

## real one
eval_every_step = 1090 ## evaluate every 10% of the total steps (10902 as seen on screen).

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
        targets = examples['label_str']

        ## truncate input and target 

        inputs = tokenizer.decode(
                    tokenizer.encode(
                        inputs,truncation=True, max_length = 900),
                    skip_special_tokens=True)
        
        examples['text'] = '<s>[INST]predict news category of the given news article below\n\n###news article\n\n {} \n\n[/INST]###news category: {} </s>'.format(inputs, targets)

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

    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold = 0.01)]
    )


    trainer.train()


def train_enc_model():

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    global dataset

    # tokenize the examples
    def convert_to_features(example_batch):
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], pad_to_max_length=True, max_length=512)

        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings

    dataset = dataset.map(convert_to_features, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=13,
        id2label=idx2label,
        label2id=label2idx
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        tokenizer=tokenizer,
        args=training_args,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold = 0.01)]
    )

    trainer.train(resume_from_checkpoint=True)


#%%

if model_name_arg in ['llama2-7b', 'mistral-7b']:
    print('training LLM')
    train_LLM()

else:
    print('training encoder-decoder pre-trained model')
    train_enc_model()