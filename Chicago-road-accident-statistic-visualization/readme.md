# Chicago-road-accident-statistic-visualization


## Overview

 
This directory contains script to run R script for visaulizing the road accident in Chicago in 2023. More details can be found in this [link](https://sites.google.com/view/chanathip-pornprasit/data-science-portfolio/chicago-road-accident-visualization).


## How to create visualization
Please follow the steps below to run script to create visualization:
1. Get the raw dataset from this [link](https://zenodo.org/records/13787952) and put it to `dataset/raw/`.
2. Run `clean_data.ipynb` to clean data
3. Run the below script files 
	* `data-analysis-person-data.R`: generate visualization of statistics of people involved in accidents 
	* `data-analysis-incident-data.R`: generate visualization of statistics of crash incidents in accidents
	* `data-analysis-combined-data.R`: generate visualization of the relationship between people and crash incidents in accidents