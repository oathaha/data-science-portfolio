library(tidyverse)

options(scipen=1000000)
theme_set(theme_bw())

df = read.csv('../dataset/cleaned/cleaned_person_data.csv')

fig.dir = '../figure/'

df = df %>%
  rename(
    Sex = SEX,
    Age = AGE
  ) %>%
  mutate(Sex = recode(Sex,
                      'F' = 'Female',
                      'M' = 'Male',
                      'X' = 'Other'
    )
  )