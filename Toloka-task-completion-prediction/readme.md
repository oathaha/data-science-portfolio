# Toloka-task-completion-prediction

## Overview

This directory contains script to run experiment for predicting time to complete an assignment in Toloka platform. More detail can be found in this [link](https://sites.google.com/view/chanathip-pornprasit/data-science-portfolio/assignment-completion-time-prediction).

## How to Replicate the Experimental Results

To replicate the experimental results, please follow the steps below
1. Get the raw dataset from this [link](https://zenodo.org/records/13787952) and put it to `dataset/raw/`.
2. Go to `./script/` directory
3. Run `clean-data.ipynb` to clean the raw dataset.
4. Run `prepare-data.py` to prepare data for model training.
5. Run `select_important_features.py` to select top-10 features for training models.
6. If you want to explore the cleaned dataset, you can run code in `explore-data.Rmd`. Note that it takes some time to render visualization due to a large dataset.
7. Run `train-model.py` to train models. For this script, the argument `--use_selected_feature` indicates that models are trained by using selected features in 5.
8. Run `evaluate.ipynb` to get evaluation results.

## Experimental Setup

In the experiment, I use grid search to search for the best hyper-parameters for all models. The details of hyper-parameters used for grid search are as follows.

|Model| Hyper-Parameter Search Space | Note
|--|--| -- |
| Lasso/Ridge | Alpha = [1, 5, 10] <br> Max_iter = [100, 500, 1000] | -- |
| ElasticNet | Alpha = [1, 5, 10] <br> Max_iter = [100, 500, 1000] <br> l1_ratio = [0.3, 0.5, 0.7, 1.0] | -- |
| AdaBoost + Lasso/Ridge  <br> AdaBoost + ElasticNet <br> AdaBoost + Linear Regression | n_estimator = [10, 50, 100] <br> learning_rate = [0.1, 0.5, 1.0, 5.0] <br> loss_func = ['linear', 'square'] | use the best hyper-parameter of Lasso/Ridge and ElasticNet to initialize base model |
| Bagging + Lasso/Ridge  <br> Bagging + ElasticNet <br> Bagging + Linear Regression | n_estimator = [10, 50, 100] | use the best hyper-parameter of Lasso/Ridge and ElasticNet to initialize base model  |


## Result

<table class="tg"><thead>
  <tr>
    <th>Model</th>
    <th>Boosting Algorithm</th>
    <th>R<sup>2</sup></th>
    <th>MSE</th>
    <th>MAE</th>
  </tr></thead>
<tbody>
    <tr>
        <td rowspan = "3">Ridge</td>
        <td>-</td>
        <td>0.29</td>
        <td>1.545</td>
        <td>0.900</td>
      </tr>
      <tr>
        <td>AdaBoost</td>
        <td>0.3</td>
        <td>1.516</td>
        <td>0.898</td>
      </tr>
      <tr>
        <td>Bagging</td>
        <td>0.29</td>
        <td>1.545</td>
        <td>0.900</td>
      </tr>
  <tr>
    <td rowspan = "3">Lasso</td>
    <td>-</td>
    <td>-0.00</td>
    <td>2.167</td>
    <td>1.158</td>
  </tr>
  <tr>
    <td>AdaBoost</td>
    <td>-0.00</td>
    <td>2.179</td>
    <td>1.182</td>
  </tr>
  <tr>
    <td>Bagging</td>
    <td>-0.00</td>
    <td>2.167</td>
    <td>1.158</td>
  </tr>
  <tr>
    <td rowspan = "3">ElasticNet</td>
    <td>-</td>
    <td>0.1</td>
    <td>1.955</td>
    <td>1.094</td>
  </tr>
  <tr>
    <td>AdaBoost</td>
    <td>0.1</td>
    <td>1.955</td>
    <td>1.094</td>
  </tr>
  <tr>
    <td>Bagging</td>
    <td>0.1</td>
    <td>1.946</td>
    <td>1.094</td>
  </tr>
    <tr>
    <td>Ensemble (average)</td>
    <td>-</td>
    <td>0.19</sup></td>
    <td>1.755</td>
    <td>1.026</td>
  </tr>
  <tr>
    <td rowspan = "3">LinearRegression</td>
    <td>-</td>
    <td>-7.34 x 10<sup>17</sup></td>
    <td>1.59 x 10<sup>18</sup></td>
    <td>7.98 x 10<sup>5</sup></td>
  </tr>
  <tr>
    <td>AdaBoost</td>
    <td>0.24</td>
    <td>1.66</td>
    <td>1.02</td>
  </tr>
  <tr>
    <td>Bagging</td>
    <td>-4.09 x 10<sup>14</sup></td>
    <td>8.85 x 10<sup>14</sup></td>
    <td>1.88 x 10<sup>4</sup></td>
  </tr>

</tbody></table>


The results show that in terms of MSE, Adaboost using Rdige as the base model achieves the lowest, which is approximately 22% and 30% lower than ElasticNet and Lasso, respectively. In terms of MAE, Adaboost using Rdige as the base model achieves approximately 18% and 22% lower than ElasticNet and Lasso, respectively. The results indicate that Adaboost using Rdige as the base model makes the least prediction error.
In terms of R-squared, Adaboost using Rdige as the base model achieves the value of 0.29 - 0.30, which is the highest when compared to Ridge, Lasso and Ensemble model. The results indicate that Adaboost using Rdige as the base model makes the most accurate prediction. 