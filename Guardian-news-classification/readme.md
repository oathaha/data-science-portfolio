# Guardian-news-classification

## Overview

This directory contains script to run experiment for classifying types of news from [The Guardian](https://www.theguardian.com/au). More details can be found in this [link](https://sites.google.com/view/chanathip-pornprasit/data-science-portfolio/news-classification).

## How to Replicate the Experimental Results

To replicate the experimental results, please follow the steps below

### Data Preparation
1. Get the raw dataset from this [link](https://zenodo.org/records/13787952).
2. Put the downloaded dataset in `dataset/raw/`.
3. Run `prepare-data.py` to prepare data for model training.

### How to Run Script for Model Fine-Tuning
1. Run `train-model.py` to train models. For this script, there is an argument `--model_name` that can be one of the following: `'llama2-7b'`(for LLaMa-2 7b), `bert` (for BERT large cased model), and `deberta` (for deBERTa large model).
2. Run `generate-prediction.py` to get the results from the trained models. For this script, there are two arguments
	`--model_name`: the base model
	`--ckpt_dir`: the directory where a model checkpoint is saved.


### How to Run Script for LLaMa-2 Prompting
1. Run `get-samples-for-few-shot-learning.py` to prepare dataset for few-shot prompting with LLaMa-2
2. Run `prompting-with-llama.py` to perform zero-shot or few-shot prompting with LLaMa-2. For this script, the argument `--prompting_technique` is required. This argument can be either `zero-shot` for zero-shot prompting or `few-shot` for few-shot prompting.

### How to Run Script to evaluate Results
Run `evaluate_result` to see evaluation results.

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
|patience (when to stop model training if validation loss does not decrease)|3|
|early stopping threshold|0.01|
|max input length|4000 (for LLaMa-2) <br> 512 (for BERT and deBERTa)|

For few-shot prompting, I use 3 examples from a training set for each testing sample.


## Result

<table>
	<thead>
	<tr>
		<th>Model</th>
		<th>Macro Precision</th>
		<th>Macro Recall</th>
		<th>Macro F1</th>
		<th>MCC</th>
	</tr></thead>
	<tbody>
	<tr>
		<td>BERT</td>
		<td>0.87</td>
		<td>0.87</td>
		<td>0.87</td>
		<td>0.87</td>
	</tr>
	<tr>
		<td>DeBERTa</td>
		<td>0.87</td>
		<td>0.87</td>
		<td>0.87</td>
		<td>0.87</td>
	</tr>
	<tr>
		<td>LLaMa-2 (Fine-tuned)</td>
		<td>0.76</td>
		<td>0.56</td>
		<td>0.60</td>
		<td>0.64</td>
	</tr>
	<tr>
		<td>LLaMa-2 (Few-shot)</td>
		<td>0.55</td>
		<td>0.55</td>
		<td>0.54</td>
		<td>0.62</td>
	</tr>
	<tr>
		<td>LLaMa-2 (Zero-shot)</td>
		<td>0.12</td>
		<td>0.09</td>
		<td>0.09</td>
		<td>0.03</td>
	</tr>
	<tr>
		<td>Baseline (weighted random)</td>
		<td>0.08</td>
		<td>0.08</td>
		<td>0.08</td>
		<td>0.06</td>
	</tr>
	</tbody>
</table>

The results show that BERT and DeBERTa achieve the same results. In particular, both achieve macro average precision, recall, and F1 of 0.87, which is 14.47%, 55.36% and 45% higher than fine-tuned LLaMa-2. In addition, both achieve MCC of 0.87, which is 35.94% higher than fine-tuned LLaMa-2. The results imply that encoder-only models can make predictions better than decoder-only models even though encoder-only models have less number of parameters than decoder-only models.

More detailed results can be found in this [link](https://sites.google.com/view/chanathip-pornprasit/data-science-portfolio/news-classification)