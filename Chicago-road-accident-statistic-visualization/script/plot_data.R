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

df = df %>% 
  mutate(Speed.limit.range = case_when(
    POSTED_SPEED_LIMIT <= 20 ~ '< 20',
    POSTED_SPEED_LIMIT > 20 & POSTED_SPEED_LIMIT <= 40 ~ '21-40',
    POSTED_SPEED_LIMIT > 40 & POSTED_SPEED_LIMIT <= 60 ~ '41-60',
    POSTED_SPEED_LIMIT > 60 & POSTED_SPEED_LIMIT <= 80 ~ '61-80',
    POSTED_SPEED_LIMIT > 80  ~ '80+',
  ))

df$Speed.limit.range = factor(df$Speed.limit.range, levels=c('< 20', '21-40', '41-60', '61-80', '80+'))

df = df %>%
  mutate(WORK_ZONE_TYPE = if_else(
    WORK_ZONE_TYPE == 'UNKNOWN', 'NOT-AT-WORK-ZONE', WORK_ZONE_TYPE
  ))

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


## detailed visualization

### number of accidents by speed limits

new.df = df %>% 
  count(Month, Speed.limit.range) %>% 
  mutate(n = log10(n))

new.df %>% ggplot(aes(x=Month, y=n, group=Speed.limit.range, color = Speed.limit.range)) + 
  geom_line() +
  geom_point() +
  labs(
    title = 'Number of accidents in each month by speed limit range',
    y = 'Count',
    color = 'Speed Limit Range'
  )

save.fig('num_accidents_each_month_by_spd_lim')


### number of accidents by weather condition

new.df = df %>% 
  count(Month, WEATHER_CONDITION) %>%
  mutate(n = log10(n))

new.df %>% ggplot(aes(x=Month, y=n, group=WEATHER_CONDITION, color = WEATHER_CONDITION)) + 
  geom_line() +
  geom_point() +
  labs(
    title = 'Number of accidents in each month by weather condition',
    y = 'Count (log base 10)',
    color = 'Weather Condition'
  ) + 
  theme(legend.position = 'bottom')

save.fig('num_accidents_each_month_by_weather')


### number of accidents by crash type

new.df = df %>% 
  count(Month, CRASH_TYPE) 
  # mutate(n = log10(n))

new.df %>% ggplot(aes(x=Month, y=n, group=CRASH_TYPE, color = CRASH_TYPE)) + 
  geom_line() +
  geom_point() +
  labs(
    title = 'Number of accidents in each month by crash type',
    y = 'Count',
    color = 'Crash Type'
  ) + 
  theme(legend.position = 'bottom') + 
  guides(color = guide_legend(nrow=2))

save.fig('num_accidents_each_month_by_crash_type')


### number of accidents by work zone type

new.df = df %>% 
  count(Month, WORK_ZONE_TYPE) %>%
  mutate(n = log10(n))

new.df %>% ggplot(aes(x=Month, y=n, group=WORK_ZONE_TYPE, color = WORK_ZONE_TYPE)) + 
  geom_line() +
  geom_point() +
  labs(
    title = 'Number of accidents in each month by type of work zone',
    y = 'Count (log base 10)',
    color = 'Type of work zone'
  ) + 
  theme(legend.position = 'bottom') + 
  guides(color = guide_legend(nrow=2))

save.fig('num_accidents_each_month_by_work_zone')


### number of accidents by surrounding conditions 

new.df = df %>% 
  select(Month, WORK_ZONE_I,WORKERS_PRESENT_I, DOORING_I, INTERSECTION_RELATED_I, HIT_AND_RUN_I, NOT_RIGHT_OF_WAY_I) %>%
  rename(
    AT_WORK_ZONE = WORK_ZONE_I,
    WORKERS_PRESENT = WORKERS_PRESENT_I,
    DOOR_OPEN = DOORING_I,
    INTERSECTION_INVOLVED = INTERSECTION_RELATED_I,
    NOT_RIGHT_OF_WAY_INVOLVED = NOT_RIGHT_OF_WAY_I,
    HIT_AND_RUN = HIT_AND_RUN_I
  ) %>% 
  pivot_longer(!Month, names_to='Surrounding_Condition') %>%
  mutate(value = case_when(
    value == 'Y' ~ 'Yes',
    value == 'N' ~ 'No',
    value == 'UNKNOWN' ~ 'UNKNOWN'
    )
  )

new.df = new.df %>% 
  filter(value == 'Yes') %>%
  count(Month, Surrounding_Condition, value) %>%
  mutate(
    # n = round(log10(n),2),
    value = factor(value, levels=c('Yes', 'No', 'UNKNOWN'))
  ) 

new.df %>% ggplot(aes(x=Month, y=n, group=Surrounding_Condition, color = Surrounding_Condition)) + 
  geom_line() +
  geom_point() +
  labs(
    title = 'Number of accidents in each month based on different surrounding conditions',
    y = 'Count',
    color = 'Surrounding conditions'
  )


save.fig('num_accidents_by_month_and_conditions')


### number of accidents based on relationship between INJURY_CLASSIFICATION and safety equipement, categorized by gender and person type 

### WILL PRESENT TABLE OF ALL TYPES LATER...
new.df = df %>%
  select(Month, INJURY_CLASSIFICATION, SAFETY_EQUIPMENT, Sex) %>%
  mutate(SAFETY_EQUIPMENT = factor(SAFETY_EQUIPMENT)) %>%
  mutate(SAFETY_EQUIPMENT = fct_recode(SAFETY_EQUIPMENT,
         "UNKNOWN" = "USAGE UNKNOWN",
         "UNKNOWN" = "UNKNOWN",
         "SAFETY BELT" = "SAFETY BELT USED",
         "NONE" = "SAFETY BELT NOT USED",
         "NONE" = "NONE PRESENT",
         "NONE" = "HELMET NOT USED",
         "HELMET" = "BICYCLE HELMET (PEDACYCLIST INVOLVED ONLY)",
         "HELMET" = "HELMET USED",
         "HELMET" = "DOT COMPLIANT MOTORCYCLE HELMET",
         "HELMET" = "NOT DOT COMPLIANT MOTORCYCLE HELMET",
         "CHILD RESTRAINT" = "CHILD RESTRAINT - REAR FACING",
         "CHILD RESTRAINT" = "CHILD RESTRAINT - TYPE UNKNOWN",
         "CHILD RESTRAINT" = "CHILD RESTRAINT - FORWARD FACING",
         "NONE" = "CHILD RESTRAINT NOT USED",
         "BOOSTER SEAT" = "BOOSTER SEAT",
         "IMPROPER EQUIPEMENT USED" = "SHOULD/LAP BELT USED IMPROPERLY",
         "IMPROPER EQUIPEMENT USED" = "CHILD RESTRAINT USED IMPROPERLY",
         "WHEELCHAIR" = "WHEELCHAIR",
         "STRETCHER" = "STRETCHER"
         )
  )



new.df = new.df %>%
  count(INJURY_CLASSIFICATION, Sex, SAFETY_EQUIPMENT) %>%
  mutate(n = log10(n))

new.df = new.df %>% 
  mutate(
    INJURY_CLASSIFICATION = fct_recode(INJURY_CLASSIFICATION,
    'FATAL' = 'FATAL',
    'INCAPACITATING' = 'INCAPACITATING INJURY',
    'NO INJURY' = 'NO INDICATION OF INJURY',
    'NONINCAPACITATING' = 'NONINCAPACITATING INJURY',
    'NOT EVIDENT' = 'REPORTED, NOT EVIDENT',
    'UNKNOWN' = 'UNKNOWN'
    )
  )

new.df %>% ggplot(aes(x=Sex, y=n, fill = SAFETY_EQUIPMENT)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(vars(INJURY_CLASSIFICATION), nrow = 2) + 
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(angle=15)
    ) +
  labs(
    title = 'Number of accidents for each injury type categorized by sefety equipement',
    x = 'Sex',
    y = 'Count (log base 10)',
    fill = 'Safety Equipment'
    ) + 
  guides(fill = guide_legend(nrow=2))

save.fig('num_accidents_by_safety_equipement_and_injury_type')
