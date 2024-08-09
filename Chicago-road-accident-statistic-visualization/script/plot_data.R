library(tidyverse)

options(scipen=1000000)
theme_set(theme_bw())

df = read.csv('../dataset/cleaned/cleaned_data.csv')

fig.dir = '../figure/'

df = df %>% mutate(CRASH_MONTH = month.abb[CRASH_MONTH])
df$CRASH_MONTH = factor(df$CRASH_MONTH, levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))

df = df %>% 
  rename(Month = CRASH_MONTH) %>%
  rename(day_of_week = CRASH_DAY_OF_WEEK) %>%
  rename(Hour = CRASH_HOUR) %>%
  rename(Sex = SEX) %>%
  rename(Age = AGE)

df = df %>%
  mutate(day_of_week = wday(day_of_week, label = T))

df = df %>%
  mutate(Sex = recode(Sex,
    'Female' = 'F',
    'Male' = 'M',
    'Other' = 'X'
  ))


df = df %>% mutate(
  Age.range = case_when(
    Age  == 'UNKNOWN' ~ 'UNKNOWN',
    Age <= 10 ~ '<=10',
    Age > 10 & Age <= 20 ~ '11-20',
    Age > 20 & Age <= 30 ~ '21-30',
    Age > 30 & Age <= 40 ~ '31-40',
    Age > 40 & Age <= 50 ~ '41-50',
    Age > 50 & Age <= 60 ~ '51-60',
    Age > 60 ~ '60+'
  )
)

df$Age.range = factor(df$Age.range, levels=c('<=10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+', 'UNKNOWN'))

save.fig = function(fig.name){
  ggsave(paste0(fig.dir, fig.name,'.png'))
}

## plot overview of data

### number of accidents by month

new.df = df %>% 
  group_by(Month) %>% 
  count()

new.df %>% ggplot(aes(Month, n, group=1)) + 
  geom_point() + 
  geom_line() + 
  labs(
    title = 'Number of accidents in each month',
    y = 'Count') +

save.fig('num_accidents_each_month')


### number of accidents by day of week

new.df = df %>%
  group_by(day_of_week) %>%
  count()

new.df %>% ggplot(aes(x=day_of_week, y=n, fill=day_of_week)) + 
  geom_bar(stat = 'identity') + 
  labs(
    title = 'Number of accidents by day of week',
    y = 'Count'
  ) + 
  theme(legend.position = 'none')

save.fig('num_accidents_by_day_of_week')


### number of accidents by hours

new.df = df %>%
  group_by(Hour) %>%
  count()

new.df %>% ggplot(aes(x=Hour, y=n, fill = Hour)) + 
  geom_bar(stat = 'identity') + 
  labs(
    title = 'Number of accidents by hour',
    y = 'Count'
  ) + 
  theme(legend.position = 'none')

save.fig('num_accidents_by_hour')


### number of accidents by gender

new.df = df %>%
  group_by(Sex) %>%
  count()

new.df %>% ggplot(aes(x=reorder(Sex, -n), y=n, fill=Sex)) +
  geom_bar(stat = 'identity') +
  labs(
    title = 'Number of accidents by gender',
    x = 'Gender', y = 'Count'
  ) +
  theme(legend.position = 'none')

save.fig('num_accidents_by_gender')


### number of accidents by age

new.df = df %>%
  group_by(Age.range) %>%
  count()


new.df %>% ggplot(aes(x=Age.range,y=n, fill=Age.range)) +
  geom_bar(stat="identity") +
  theme(legend.position = 'none') +
  labs(
    title = 'Number of accidents by age range',
    x = 'Age Range',
    y = 'Count'
  )

save.fig('num_accident_by_age_range')


## number of accidents by weather conditions

new.df = df %>%
  count(WEATHER_CONDITION) %>% 
  mutate(n = round(log10(n),2))
  
new.df %>% ggplot(aes(y=reorder(WEATHER_CONDITION, n), x=n, fill=WEATHER_CONDITION)) +
  geom_bar(stat = 'identity') + 
  geom_text(aes(label=n), position=position_dodge(width=0.9), hjust=0.65) +
  theme(legend.position = 'none') +
  labs(
    title = 'Number of accidents by weather condition',
    x = 'Count (scale log base 10)',
    y = 'Weather Condition'
  )

save.fig('num_accident_by_weather_cond')


## number of accidents by prime cause

new.df = df %>%
  count(PRIM_CONTRIBUTORY_CAUSE) %>%
  arrange(desc(n)) %>%
  head(10)

new.df %>% ggplot(aes(y=reorder(PRIM_CONTRIBUTORY_CAUSE, n), x=n, fill=PRIM_CONTRIBUTORY_CAUSE)) +
  geom_bar(stat = 'identity') +
  theme(legend.position = 'none') +
  geom_text(aes(label=n, hjust = 0.7)) +
  labs(
    title = 'Top-10 causes of accidents', 
    x = 'Count',
    y = 'Primary Contribution Cause'
  )

save.fig('top_10_accidents_causes')


## top-10 states that have the highest accidents

new.df = df %>%
  count(STATE) %>%
  mutate(n = log10(n)) %>%
  arrange(desc(n)) %>%
  head(10)

new.df %>% ggplot(aes(x=reorder(STATE, -n), y=n, fill=STATE)) + 
  geom_bar(stat = 'identity') + 
  theme(legend.position = 'none') + 
  labs(
    title = 'Top-10 states with the highest number of accidents', 
    x = 'State',
    y = 'Count (log base 10)'
  )

save.fig('top-10 states with accidents')
