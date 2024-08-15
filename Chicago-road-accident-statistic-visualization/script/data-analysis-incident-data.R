library(tidyverse)
library(ggmosaic)

options(scipen=1000000)
theme_set(theme_bw())

df = read.csv('../dataset/cleaned/cleaned_incident_data.csv')

fig.dir = '../figure/'

df = df %>% 
  rename(
    Month = CRASH_MONTH,
    day_of_week = CRASH_DAY_OF_WEEK
  ) %>%
  mutate(
    Month = month.abb[Month],
    Month = factor(Month, levels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")),
    day_of_week = wday(day_of_week, label = T),
    Speed.limit.range = case_when(
      POSTED_SPEED_LIMIT <= 20 ~ '< 20',
      POSTED_SPEED_LIMIT > 20 & POSTED_SPEED_LIMIT <= 40 ~ '21-40',
      POSTED_SPEED_LIMIT > 40 & POSTED_SPEED_LIMIT <= 60 ~ '41-60',
      POSTED_SPEED_LIMIT > 60 & POSTED_SPEED_LIMIT <= 80 ~ '61-80',
      POSTED_SPEED_LIMIT > 80  ~ '80+',
    ),
    Speed.limit.range = factor(Speed.limit.range, levels=c('< 20', '21-40', '41-60', '61-80', '80+')),
    WORK_ZONE_TYPE = if_else(
      WORK_ZONE_TYPE == 'UNKNOWN', 'NOT-AT-WORK-ZONE', WORK_ZONE_TYPE),
    INJURIES_FATAL = as.numeric(str_replace(INJURIES_FATAL, 'UNKNOWN','0')),
        INJURIES_INCAPACITATING = as.numeric(str_replace(INJURIES_INCAPACITATING, 'UNKNOWN','0')),
        INJURIES_NON_INCAPACITATING = as.numeric(str_replace(INJURIES_NON_INCAPACITATING, 'UNKNOWN','0')),
        INJURIES_REPORTED_NOT_EVIDENT = as.numeric(str_replace(INJURIES_REPORTED_NOT_EVIDENT, 'UNKNOWN','0'))
  )
  

save.fig = function(fig.name, scale=1){
  ggsave(paste0(fig.dir, fig.name,'.png'), scale=scale)
}



## num accidents by weather and month

new.df = df %>% 
  filter(WEATHER_CONDITION != 'UNKNOWN') %>%
  count(Month, WEATHER_CONDITION)
  # mutate(n = log10(n))

new.df %>% ggplot(aes(x=Month, y=WEATHER_CONDITION, fill=n)) + 
  geom_tile() + 
  theme(axis.text.x = element_text(angle=15, vjust = 0.5)) +
  labs(
    title = 'Number of accident in each month by weather condition',
    y = 'Weather Condition',
    fill = 'Number of Accident'
  )
  # theme(legend.position = 'bottom')

save.fig('num_accidents_each_month_by_weather')



## num accidents by day of week and month

new.df = df %>%
  group_by(Month, day_of_week) %>%
  count()

new.df %>% ggplot(aes(x=Month, y=n, color=day_of_week, group=day_of_week)) + 
  geom_line() +
  geom_point() +
  labs(
    title = 'Number of accidents in each month by day of week',
    y = 'Number of Accident'
  ) + 
  theme(legend.position = 'bottom') +
  scale_color_manual(values = c('red', 'yellow', 'pink', 'green', 'orange', 'blue', 'purple'))

save.fig('num_accidents_by_day_of_week')



## top-10 contribution cause

new.df = df %>%
  count(PRIM_CONTRIBUTORY_CAUSE) %>%
  arrange(desc(n)) %>%
  head(10)

top.10.cause = new.df$PRIM_CONTRIBUTORY_CAUSE

new.df %>% ggplot(aes(y=reorder(PRIM_CONTRIBUTORY_CAUSE, n), x=n, fill=PRIM_CONTRIBUTORY_CAUSE)) +
  geom_bar(stat = 'identity') +
  theme(legend.position = 'none') +
  geom_text(aes(label=n, hjust = 0.7)) +
  labs(
    title = 'Top-10 causes of accidents', 
    x = 'Number of Accident',
    y = 'Primary Contribution Cause'
  )

save.fig('top_10_accidents_causes')



## top-10 first crash type

new.df = df %>%
  count(FIRST_CRASH_TYPE) %>%
  arrange(desc(n)) %>%
  head(10)

top.10.crash.type = new.df$FIRST_CRASH_TYPE

new.df %>% ggplot(aes(y=reorder(FIRST_CRASH_TYPE, n), x=n, fill=FIRST_CRASH_TYPE)) +
  geom_bar(stat = 'identity') +
  theme(legend.position = 'none') +
  geom_text(aes(label=n, hjust = 0.7)) +
  labs(
    title = 'Top-10 crash type in accident', 
    x = 'Number of Accident',
    y = 'Crash Type'
  )

save.fig('top_10_crash_type')



## number of different injury types in each month

new.df = df %>%
  select(INJURIES_FATAL, INJURIES_INCAPACITATING, INJURIES_NON_INCAPACITATING, INJURIES_REPORTED_NOT_EVIDENT, Month) %>%
  group_by(Month) %>%
  summarise(across(everything(), sum))

new.df = new.df %>%
  pivot_longer(
    cols = starts_with('INJ'),
    names_to = 'Injury_Type'
      ) %>%
  mutate(Injury_Type = str_replace(Injury_Type, 'INJURIES_',''))

new.df %>% ggplot(aes(x=Month, y = value, color = Injury_Type, group = Injury_Type)) + 
  geom_line() + 
  geom_point() +
  labs(
    title = 'Number of people at crash site in each month by injury type',
    y = 'Number of People',
    color = 'Injury Type'
  )

save.fig('num_injured_people_each_month_by_injury_type')



## number of accidents in each month by surroundings in crash site

new.df = df %>%
  select(INTERSECTION_RELATED_I, NOT_RIGHT_OF_WAY_I, HIT_AND_RUN_I, DOORING_I, WORK_ZONE_I, Month) %>%
  pivot_longer(!Month, names_to='Surrounding_Condition') %>%
  filter(value == 'Y') %>%
  count(Month, Surrounding_Condition) %>%
  mutate(Surrounding_Condition = str_replace(Surrounding_Condition, '_I',''))

new.df %>% ggplot(aes(x=Month, y = n, color = Surrounding_Condition, group = Surrounding_Condition)) + 
  geom_line() + 
  geom_point() +
  labs(
    title = 'Total accident in each month by surrounding situation',
    y = 'Number of Accident',
    color = 'Surrounding Situation'
  )

save.fig('num_accidents_each_month_by_situation')



## number of accidents by crash type and roadway surface condition

new.df = df %>%
  filter(ROADWAY_SURFACE_COND != 'UNKNOWN') %>%
  # count(FIRST_CRASH_TYPE, ROADWAY_SURFACE_COND) %>%
  mutate(
    ROADWAY_SURFACE_COND = factor(ROADWAY_SURFACE_COND, levels=c('DRY', 'ICE', 'WET', 'SAND, MUD, DIRT', 'SNOW OR SLUSH', 'OTHER'))
  ) %>%
  select(ROADWAY_SURFACE_COND, FIRST_CRASH_TYPE)

new.df %>% ggplot() + 
  geom_mosaic(aes(x=product(FIRST_CRASH_TYPE,ROADWAY_SURFACE_COND), fill=FIRST_CRASH_TYPE), offset = 0.03) + 
  theme(
    legend.position = 'none',
    axis.text.x = element_text(
      angle = 90, 
      vjust=0.5,
      # size = 20
    ),
    # axis.text.y = element_text(size = 20),
    # plot.title = element_text(size = 40)
  ) + 
  labs(
    title = 'Number of accident by crash type and roadway surface condition',
    x = 'Roadway Surface Condition',
    y = 'Crash Type'
    # fill = 'Driver Action'
  )

# new.df %>% ggplot(aes(x=n, y = reorder(FIRST_CRASH_TYPE, n), fill=ROADWAY_SURFACE_COND)) + 
#   geom_bar(stat = 'identity', position = 'fill') + 
#   labs(
#     title = 'Number of accident by crash type and roadway surface condition',
#     x = 'Ratio',
#     y = 'Crash Type',
#     fill = 'Roadway Surface Condition'
#   )

save.fig('num_accident_by_crash_type_and_roadway_surface', scale=1.5)  



## number of different road defect in each month

new.df = df %>%
  select(INJURIES_FATAL, INJURIES_INCAPACITATING, INJURIES_NON_INCAPACITATING, INJURIES_REPORTED_NOT_EVIDENT, ROAD_DEFECT) %>%
  filter(ROAD_DEFECT != 'UNKNOWN') %>%
  group_by(ROAD_DEFECT) %>%
  summarise(across(everything(), sum))

new.df = new.df %>%
  pivot_longer(
    cols = starts_with('INJ'),
    names_to = 'Injury_Type'
  ) %>%
  mutate(Injury_Type = str_replace(Injury_Type, 'INJURIES_',''))

new.df %>% ggplot(aes(x=value, y = reorder(ROAD_DEFECT, value), fill=Injury_Type)) + 
  geom_bar(stat = 'identity', position = 'fill') + 
  labs(
    title = 'Number of accident by crash type and roadway surface condition',
    x = 'Ratio',
    y = 'Crash Type',
    fill = 'Roadway Surface Condition'
  ) + 
  theme(legend.position = 'bottom')

save.fig('num_injured_people_each_month_by_injury_type')

