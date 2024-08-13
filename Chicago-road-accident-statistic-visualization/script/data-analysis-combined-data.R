library(tidyverse)
library(ggsankey)

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
    'F' = 'Female',
    'M' = 'Male',
    'X' = 'Other'
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

df$INJURY_CLASSIFICATION = factor(df$INJURY_CLASSIFICATION)

df = df %>% 
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


save.fig = function(fig.name, scale=1){
  ggsave(paste0(fig.dir, fig.name,'.png'), scale=scale)
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


### number of accidents in each month by injury types
df %>% 
  count(Month, INJURY_CLASSIFICATION) %>% 
  mutate(n = log10(n)) %>%
  ggplot(aes(x=Month, y=n, color = INJURY_CLASSIFICATION, group = INJURY_CLASSIFICATION)) + 
  geom_line() + 
  geom_point() + 
  labs(
    title = 'Number of accidents in each month by injury types',
    y = 'Count (log base 10)',
    color = 'Injury Type'
  )

save.fig('num_accident_by_month_and_injury_type')


### number of accidents based on relationship between INJURY_CLASSIFICATION and safety equipement, categorized by gender and person type 


### WILL PRESENT TABLE OF ALL TYPES LATER...
new.df = df %>%
  select(Month, INJURY_CLASSIFICATION, SAFETY_EQUIPMENT, Sex) %>%
  filter(INJURY_CLASSIFICATION != "NO INJURY") %>%
  mutate(SAFETY_EQUIPMENT = factor(SAFETY_EQUIPMENT))  %>%
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


new.df %>% ggplot(aes(x=Sex, y=n, fill = SAFETY_EQUIPMENT)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(vars(INJURY_CLASSIFICATION), nrow = 3) + 
  theme(
    # legend.position = 'bottom',
    axis.text.x = element_text(angle=15)
    ) +
  labs(
    title = 'Number of accidents for each injury type categorized by sefety equipement',
    x = 'Sex',
    y = 'Count (log base 10)',
    fill = 'Safety Equipment'
    ) 
  # guides(fill = guide_legend(nrow=2))

save.fig('num_accidents_by_safety_equipement_and_injury_type')


### number of accidents based on relationship between INJURY_CLASSIFICATION and airbag deployment, categorized by gender and person type 

### WILL PRESENT TABLE OF ALL TYPES LATER...
new.df = df %>%
  select(Month, INJURY_CLASSIFICATION, AIRBAG_DEPLOYED, Sex) %>%
  mutate(AIRBAG_DEPLOYED = recode(AIRBAG_DEPLOYED,
      "DEPLOYMENT UNKNOWN" = "UNKNOWN",
      "DID NOT DEPLOY" = "NO DEPLOYMENT",
      "NOT APPLICABLE" = "NOT APPLICABLE",
      "UNKNOWN" = "NOT APPLICABLE",
      "DEPLOYED, FRONT" = "DEPLOYED",
      "DEPLOYED, COMBINATION" = "DEPLOYED",
      "DEPLOYED, SIDE" = "DEPLOYED",   
      "DEPLOYED OTHER (KNEE, AIR, BELT, ETC.)" = "DEPLOYED"
    )
  ) 
# %>%
  # mutate(AIRBAG_DEPLOYED = factor(AIRBAG_DEPLOYED))


new.df = new.df %>%
  filter(INJURY_CLASSIFICATION != 'NO INJURY') %>%
  count(INJURY_CLASSIFICATION, Sex, AIRBAG_DEPLOYED) %>%
  mutate(n = log10(n))


new.df %>% ggplot(aes(x=Sex, y=n, fill = AIRBAG_DEPLOYED)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(vars(INJURY_CLASSIFICATION), nrow = 2) + 
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(angle=15)
  ) +
  labs(
    title = 'Number of accidents for each injury type categorized by airbag deployment',
    x = 'Sex',
    y = 'Count (log base 10)',
    fill = 'Airbag Deployment'
  ) 
  # guides(fill = guide_legend(nrow=2))

save.fig('num_accidents_by_airbag_deployment_and_injury_type')


### number of accidents based on relationship between INJURY_CLASSIFICATION and airbag deployment, categorized by gender and person type 

### WILL PRESENT TABLE OF ALL TYPES LATER...
new.df = df %>%
  select(Month, INJURY_CLASSIFICATION, EJECTION, Sex) %>%
  mutate(EJECTION = recode(EJECTION,
    "UNKNOWN" = "NONE"
    )
  ) 
# %>%
# mutate(AIRBAG_DEPLOYED = factor(AIRBAG_DEPLOYED))


new.df = new.df %>%
  filter(INJURY_CLASSIFICATION != 'NO INJURY') %>%
  count(INJURY_CLASSIFICATION, Sex, EJECTION) %>%
  mutate(n = log10(n))


new.df %>% ggplot(aes(x=Sex, y=n, fill = EJECTION)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(vars(INJURY_CLASSIFICATION), nrow = 2) + 
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(angle=15)
  ) +
  labs(
    title = 'Number of accidents for each injury type categorized by ejection',
    x = 'Sex',
    y = 'Count (log base 10)',
    fill = 'Ejection'
  ) 
# guides(fill = guide_legend(nrow=2))

save.fig('num_accidents_by_ejection_and_injury_type')



### number of accidents by relationship between crash type and driver action categorized by gender

### will create table for this one later...

new.df = df %>%
  filter(
    PERSON_TYPE == 'DRIVER' & !DRIVER_ACTION %in% c('NONE', 'UNKNOWN')
    ) %>%
  select(Month, DRIVER_ACTION, CRASH_TYPE) %>%
  count(Month, DRIVER_ACTION, CRASH_TYPE) %>%
  mutate(DRIVER_ACTION = recode(DRIVER_ACTION,
                                # "UNKNOWN" = "NONE",
                                "IMPROPER BACKING" = "IMPROPER DRIVING",
                                "TOO FAST FOR CONDITIONS" = "IMPROPER DRIVING",
                                "FOLLOWED TOO CLOSELY" = "IMPROPER DRIVING",
                                "IMPROPER PASSING" = "IMPROPER DRIVING",
                                "IMPROPER LANE CHANGE" = "IMPROPER DRIVING",
                                "WRONG WAY/SIDE" = "IMPROPER DRIVING",
                                "IMPROPER TURN" = "IMPROPER DRIVING",
                                "IMPROPER TURN" = "IMPROPER DRIVING",)
         )
  # mutate(n = log2(n))

new.df %>% ggplot(aes(x=Month, y=n, fill = DRIVER_ACTION)) + 
  geom_col(position = 'fill') + 
  facet_wrap(vars(CRASH_TYPE)) + 
  labs(
    title = "Number of accidents in each month by driver's action",
    y = 'Count',
    fill = 'Driver Action'
    ) + 
  theme(axis.text.x = element_text(angle = 30))

save.fig('num_accidents_each_month_by_driver_action')


### will create table for this one later...

new.df = df %>%
  filter(
    PERSON_TYPE == 'DRIVER' & DRIVER_VISION != 'UNKNOWN'
  ) %>%
  select(Month, DRIVER_VISION, CRASH_TYPE)  %>%
  mutate(DRIVER_VISION = recode(DRIVER_VISION,
                                "MOVING VEHICLES" = "VEHICLES",
                                "PARKED VEHICLES" = "VEHICLES",
                                "BLINDED - SUNLIGHT" = "BLINDED",
                                "BLINDED - HEADLIGHTS" = "BLINDED",
                                "IMPROPER LANE CHANGE" = "IMPROPER DRIVING",
                                "WRONG WAY/SIDE" = "IMPROPER DRIVING",
                                "IMPROPER TURN" = "IMPROPER DRIVING",
                                "IMPROPER TURN" = "IMPROPER DRIVING",)
  )%>%
  count(Month, DRIVER_VISION, CRASH_TYPE)
# mutate(n = log10(n))

new.df %>% ggplot(aes(x=Month, y=n, fill = DRIVER_VISION)) + 
  geom_col(position = 'fill') + 
  facet_wrap(vars(CRASH_TYPE)) + 
  labs(
    title = "Number of accidents in each month by driver's action",
    y = 'Count',
    fill = 'Driver Action'
  ) + 
  theme(axis.text.x = element_text(angle = 30))

save.fig('num_accidents_each_month_by_driver_vision')



### number of accidents by injury types and physical conditions

new.df = df %>%
  filter(PERSON_TYPE == 'DRIVER') %>%
  count(INJURY_CLASSIFICATION, PHYSICAL_CONDITION, Sex) %>% 
  mutate(n = log10(n))

new.df %>% ggplot(aes(x=n, y=INJURY_CLASSIFICATION, fill = PHYSICAL_CONDITION)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  scale_fill_brewer(palette="Paired") +
  facet_wrap(vars(Sex), nrow=2) +
  theme(legend.position = 'bottom') + 
  labs(
    title = 'Number of accidents by physical condition and injury type',
    x = 'Count (log base 10)', 
    y = 'Injury Type',
    fill = 'Physical Condition'
  )

save.fig('num_accidents_by_physical_condition_and_injury_type')


### blood alc concentration result by person type

new.df = df %>%
  count(BAC_RESULT, PERSON_TYPE, Sex) %>%
  mutate(
    n = log10(n),
  )

new.df %>% ggplot(aes(x=n, y=PERSON_TYPE, fill=BAC_RESULT)) +
  geom_bar(stat='identity', position = 'dodge') + 
  facet_wrap(vars(Sex)) +
  theme(legend.position = 'bottom') + 
  guides(fill = guide_legend(nrow=2)) + 
  labs(
    title = 'Blood Alcohol Concentration Test Result Categorized by Person Type',
    x = 'Count (log base 10)',
    y = 'Person Type',
    fill = 'Blood Alcohol Concentration\nTest Result'
  )

save.fig('BAC_result_by_person_type')


### BAC test result based on crash type and gender

new.df = df %>%
  filter(BAC_RESULT.VALUE != 'UNKNOWN' & !Sex %in% c('UNKNOWN', 'X') & PERSON_TYPE == 'DRIVER') %>%
  select(BAC_RESULT.VALUE, CRASH_TYPE, Sex, INJURY_CLASSIFICATION) %>%
  mutate(BAC_RESULT.VALUE = as.numeric(BAC_RESULT.VALUE))

new.df %>% ggplot(aes(x=BAC_RESULT.VALUE, y=INJURY_CLASSIFICATION, color=Sex)) + 
  geom_boxplot() + 
  # facet_wrap(vars(Sex)) +
  theme(legend.position = 'bottom') + 
  labs(
    title = 'Blood Alcohol Concentration Test Result by Injury Type',
    x = 'Injury Type',
    y = 'Blood Alcohol Concentration Test Result'
  ) + 
  scale_x_continuous(breaks = seq(0.0, 1.0, 0.1))

save.fig('bac_result_by_injury_type')



### number of cell phone usage by gender and age range

new.df = df %>%
  filter(CELL_PHONE_USE != 'UNKNOWN' & 
           Sex != 'UNKNOWN' & 
           Age.range != 'UNKNOWN') %>%
  count(CELL_PHONE_USE, Sex, Age.range)

new.df %>% ggplot(aes(x=n, y=Age.range, fill = CELL_PHONE_USE)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(vars(Sex)) + 
  labs(
    title = 'Cell Phone Usage based on Age and Gender',
    x = 'Count',
    y = 'Age Range',
    fill = 'Cell Phone Usage'
  )

save.fig('cell_phone_usage')


### number of accidents by PEDPEDAL_ACTION and injury type

new.df = df %>%
  filter(
    !PEDPEDAL_ACTION %in% c('UNKNOWN',"UNKNOWN/NA") & 
      INJURY_CLASSIFICATION != 'UNKNOWN' & 
      PERSON_TYPE %in% c("PEDESTRIAN", "BICYCLE")
    ) %>%
  select(PEDPEDAL_ACTION, INJURY_CLASSIFICATION) %>%
  mutate(PEDPEDAL_ACTION_TYPE = recode(PEDPEDAL_ACTION,
                                       "CROSSING - NO CONTROLS (AT INTERSECTION)" = 'CROSSING',
                                       "CROSSING - NO CONTROLS (NOT AT INTERSECTION)" = 'CROSSING',
                                       "CROSSING - WITH SIGNAL" = "CROSSING",
                                       "CROSSING - AGAINST SIGNAL" = "CROSSING",
                                       "TURNING RIGHT" = "CHANGE DIRECTION",
                                       "STANDING IN ROADWAY" = "IN ROADWAY",
                                       "TURNING RIGHT" = "CHANGE DIRECTION",
                                       "PLAYING IN ROADWAY" = "IN ROADWAY",
                                       "WORKING IN ROADWAY" = "IN ROADWAY",
                                       "CROSSING - CONTROLS PRESENT (NOT AT INTERSECTION)" = "CROSSING",
                                       "TO/FROM DISABLED VEHICLE" = "IN VEHICLE",
                                       "PLAYING/WORKING ON VEHICLE" = "IN VEHICLE"
  ))
  # count(PEDPEDAL_ACTION, INJURY_CLASSIFICATION, PEDPEDAL_ACTION_TYPE)

new.df = new.df %>%
  make_long(PEDPEDAL_ACTION_TYPE, PEDPEDAL_ACTION, INJURY_CLASSIFICATION)

dagg = new.df %>%
  group_by(node)%>%
  tally()

new.df = merge(new.df, dagg, by.x = 'node', by.y = 'node', all.x = TRUE)

new.df %>% ggplot(aes(x = x, 
                      next_x = next_x , 
                      node = node , 
                      next_node = next_node , 
                      fill = factor(node) , 
                      label = paste0(node,": ", n))) + 
  geom_sankey(flow.alpha = 0.5 , node.color = "black" ,show.legend = FALSE) + 
  geom_sankey_label(size = 6, color = "black", fill= "white") + 
  theme(
    legend.position = "none",
    axis.text.y = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(size = 20, angle = 15),
    plot.title = element_text(size=36)
    ) + 
  labs(
    title = 'Number of Injury Type by Pedestrain Actions',
    x = ''
  )

save.fig('test', scale=4.5)


### number of accidents by PEDPEDAL_VISIBILITY and injury type categorized by gender

new.df = df %>%
  filter(
      PEDPEDAL_VISIBILITY != 'UNKNOWN' & 
      LIGHTING_CONDITION != 'UNKNOWN' & 
      PERSON_TYPE %in% c("PEDESTRIAN", "BICYCLE") &
      Sex %in% c('Female', 'Male') 
  ) %>%
  count(PEDPEDAL_VISIBILITY, LIGHTING_CONDITION, Sex) %>%
  mutate(n = log2(n))

new.df %>% ggplot(aes(x=n, y=LIGHTING_CONDITION, fill = PEDPEDAL_VISIBILITY)) + 
  geom_bar(stat = 'identity', position = 'dodge') +
  facet_wrap(vars(Sex)) + 
  theme(legend.position = 'bottom') +
  guides(fill = guide_legend(nrow=2))
