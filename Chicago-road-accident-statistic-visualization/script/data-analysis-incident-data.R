library(tidyverse)

options(scipen=1000000)
theme_set(theme_bw())

df = read.csv('../dataset/cleaned/cleaned_person_data.csv')

fig.dir = '../figure/'