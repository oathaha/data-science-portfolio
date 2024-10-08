library(tidyverse)
library(ggmosaic)

options(scipen=1000000)
theme_set(theme_bw(base_size = 15))

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
        INJURIES_REPORTED_NOT_EVIDENT = as.numeric(str_replace(INJURIES_REPORTED_NOT_EVIDENT, 'UNKNOWN','0')),
    FIRST_CRASH_TYPE = recode(FIRST_CRASH_TYPE,
      "REAR TO FRONT" = "REAR",
      "REAR END" = "REAR",
      "REAR TO SIDE" = "REAR",
      "REAR TO REAR" = "REAR",
      "SIDESWIPE SAME DIRECTION" = "SIDESWIPE",
      "SIDESWIPE OPPOSITE DIRECTION" = "SIDESWIPE"
    )
  )
  

save.fig = function(fig.name, scale=1){
  ggsave(paste0(fig.dir, fig.name,'.png'), scale=scale)
}



## Number of accident in each month by weather condition

new.df = df %>% 
  filter(WEATHER_CONDITION != 'UNKNOWN') %>%
  count(Month, WEATHER_CONDITION)

new.df %>% ggplot(aes(x=Month, y=WEATHER_CONDITION, fill=n)) + 
  geom_tile() + 
  theme(axis.text.x = element_text(angle=15, vjust = 0.5)) +
  labs(
    title = str_wrap('Number of accident in each month by weather condition', width=50),
    y = 'Weather Condition',
    fill = 'Number of \nAccident'
  ) + 
  scale_y_discrete(labels = label_wrap_gen(15)) 

save.fig('num_accidents_each_month_by_weather', 1.3)



## Number of accidents in each month by day of week

new.df = df %>%
  group_by(Month, day_of_week) %>%
  count()

new.df %>% ggplot(aes(x=Month, y=n, color=day_of_week, group=day_of_week)) + 
  geom_line() +
  geom_point() +
  labs(
    title = str_wrap('Number of accidents in each month by day of week', width=40),
    y = 'Number of Accident'
  ) + 
  theme(legend.position = 'bottom') +
  scale_color_manual(values = c('red', 'yellow', 'pink', 'green', 'orange', 'blue', 'purple')) + 
  expand_limits(y=0)

save.fig('num_accidents_by_day_of_week')



## Top-10 causes of accidents

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
  ) + 
  scale_y_discrete(labels = label_wrap_gen(20))

save.fig('top_10_accidents_causes', 1.3)



## Top-10 crash type in accident

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
  ) + 
  scale_y_discrete(labels = label_wrap_gen(10))

save.fig('top_10_crash_type', 1.3)



## Number of people at crash site in each month by injury type

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
    title = str_wrap('Number of people at crash site in each month by injury type', width=60),
    y = 'Number of People',
    color = 'Injury Type'
  ) + 
  theme(legend.position = 'bottom') + 
  guides(color = guide_legend(nrow=2))

save.fig('num_injured_people_each_month_by_injury_type', 1.5)



## Total accident in each month by surrounding situation

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
    title = str_wrap('Total accident in each month by surrounding situation', width=70),
    y = 'Number of Accident',
    color = 'Surrounding Situation'
  ) +
  theme(axis.text.x = element_text(angle=30))

save.fig('num_accidents_each_month_by_situation')



## Number of accident by crash type and roadway surface condition

new.df = df %>%
  filter(ROADWAY_SURFACE_COND != 'UNKNOWN') %>%

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
    ),
  ) + 
  labs(
    title = 'Number of accident by crash type and roadway surface condition',
    x = 'Roadway Surface Condition',
    y = 'Crash Type'
  )


save.fig('num_accident_by_crash_type_and_roadway_surface', scale=1.5)  



## Number of accident by crash type and roadway surface condition

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
    title = str_wrap('Number of accident by crash type and roadway surface condition', width=45),
    x = 'Ratio',
    y = 'Crash Type',
    fill = 'Roadway Surface\nCondition'
  )

save.fig('num_injured_people_by_road_defect',1.3)