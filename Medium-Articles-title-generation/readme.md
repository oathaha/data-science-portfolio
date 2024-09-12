# Medium-Articles-title-generation

## Overview

This directory contains script to run experiment for generating titles for medium articles. More details can be found in this [link](www.tmp.com).

## How to Replicate the Experimental Results

To replicate the results, please follow the steps below.


### Data Preparation
1. Get the raw dataset from this [link](https://drive.google.com/drive/folders/1w_3rMmeEpQlBlHTCqupwSBHxwFoYXsEK?usp=drive_link).
2. Put the downloaded dataset in `dataset/raw/`.
3. Run `prepare-data.py` to prepare data for model training.

### How to Run Script for Model Fine-Tuning
1. Run `train-model.py` to train models. For this script, there is an argument `--model_name` that can be one of the following: `'llama2-7b'`(for LLaMa-2 7b), `t5` (for T5 3b model), and `bart` (for BART large model).
2. Run `generate.py` to get the results from the trained models. For this script, there are two arguments
	`--model_name`: the base model
	`--ckpt_dir`: the directory where a model checkpoint is saved.


### How to Run Script for LLaMa-2 Prompting
1. Run `get-samples-for-few-shot-learning.py` to prepare dataset for few-shot prompting with LLaMa-2
2. Run `prompting-with-llama.py` to perform zero-shot or few-shot prompting with LLaMa-2. For this script, the argument `--prompting_technique` is required. This argument can be either `zero-shot` for zero-shot prompting or `few-shot` for few-shot prompting.

### How to Run Script to evaluate Results
1. Run `evaluate.py` to get evaluation results.
2. Run `show-evaluation-results.ipynb` to see evaluation results.

## Experimental Setup

In the experiment, I use the following hyper-parameter settings to fine-tune models.

|hyper-parameter| value |
|--|--|
| validation step | 3050 |
|learning rate|2e-5|
|train batch size|8|
|validation batch size|8|
|weight decay|0.01|
|validation metric|loss|
|patience (when to stop model training if validation loss does not decrease)|3|
|early stopping threshold|0.01|
|max input length|3090 (for LLaMa-2) <br> 512 (for T5 and BART)|

For few-shot prompting, I use 3 examples from a training set for each testing sample.

## Result

The results show that T5 achieves ROUGE-L of 40.32% and BLEU-4 of 27.45%, which is the highest when compared to other models. On the other hand, BART achieves METEOR of 39.12% and BERTScore of 89.21%, which is the highest when compared to other models. The results indicate that T5 and BART achieve the highest performance when compared to Long-T5 and the variation of LLaMa-2 model.

More detailed results can be found in this [link](tmp)