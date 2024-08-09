# Guardian-news-classification

## Overview

This directory contains script to run experiment for classifying types of news from [The Guardian](https://www.theguardian.com/au). More details can be found in this [link](www.tmp.com).

## How to Replicate the Experimental Results

To replicate the experimental results, please follow the steps below
1. Get the raw dataset from this [link](https://drive.google.com/drive/folders/1w_3rMmeEpQlBlHTCqupwSBHxwFoYXsEK?usp=drive_link) and put it to `dataset/raw/`.
2. Go to `./script/` directory
3. Run `prepare-data.py` to prepare data for model training.
4. Run `train-model.py` to train models. For this script, there is an argument `--model_name` that can be one of the following: `'llama2-7b'`(for LLaMa-2 7b), `bert` (for BERT large cased model), and `deberta` (for deBERTa large model).
5. Run `generate-prediction.py` to get the results from the trained models. For this script, there are two arguments
	`--model_name`: the base model
	`--ckpt_dir`: the directory where a model checkpoint is saved.

8. Run `show-evaluation-results.ipynb` to see evaluation results.

## Experimental Setup

In the experiment, I use the following hyper-parameter settings to train models.

|hyper-parameter| value |
|--|--|
| validation step | 1090 |
|learning rate|2e-5|
|train batch size|8|
|validation batch size|8|
|weight decay|0.01|
|validation metric|loss|
|patience (when to stop model training if loss does not decrease)|3|
|early stopping threshold|0.01|
|max sequence length|4000 (for LLaMa-2) <br> 512 (for BERT and deBERTa)|

## Result

later...
