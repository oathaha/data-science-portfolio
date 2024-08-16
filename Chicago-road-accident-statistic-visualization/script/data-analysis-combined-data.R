library(tidyverse)
library(ggsankey)
library(ggmosaic)

options(scipen=1000000)
theme_set(theme_bw(base_size = 15))

df = read.csv('../dataset/cleaned/cleaned_joined_data.csv')

fig.dir = '../figure/'

df = df %>%
  rename(Sex = SEX, Age = AGE, Month = CRASH_MONTH, day_of_week = CRASH_DAY_OF_WEEK) %>%
  mutate(
    Sex = recode(Sex, 'F' = 'Female', 'M' = 'Male', 'X' = 'Other'),
    Age.range = case_when(
      Age  == 'UNKNOWN' ~ 'UNKNOWN',
      Age <= 10 ~ '<= 10',
      Age > 10 & Age <= 20 ~ '11-20',
      Age > 20 & Age <= 30 ~ '21-30',
      Age > 30 & Age <= 40 ~ '31-40',
      Age > 40 & Age <= 50 ~ '41-50',
      Age > 50 & Age <= 60 ~ '51-60',
      Age > 60 ~ '60+'
    ),
   Age.range = factor(Age.range, levels=c('<= 10', '11-20', '21-30', '31-40', '41-50', '51-60', '60+', 'UNKNOWN')),
   INJURY_CLASSIFICATION = factor(INJURY_CLASSIFICATION),
   INJURY_CLASSIFICATION = fct_recode(INJURY_CLASSIFICATION,
      'FATAL' = 'FATAL',
      'INCAPACITATING' = 'INCAPACITATING INJURY',
      'NO INJURY' = 'NO INDICATION OF INJURY',
      'NONINCAPACITATING' = 'NONINCAPACITATING INJURY',
      'NOT EVIDENT' = 'REPORTED, NOT EVIDENT',
      'UNKNOWN' = 'UNKNOWN'
      ),
  SAFETY_EQUIPMENT = factor(SAFETY_EQUIPMENT),
  SAFETY_EQUIPMENT = fct_recode(SAFETY_EQUIPMENT,
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
         ),
  AIRBAG_DEPLOYED = recode(AIRBAG_DEPLOYED,
      "DEPLOYMENT UNKNOWN" = "UNKNOWN",
      "DID NOT DEPLOY" = "NO DEPLOYMENT",
      "NOT APPLICABLE" = "NOT APPLICABLE",
      "UNKNOWN" = "NOT APPLICABLE",
      "DEPLOYED, FRONT" = "DEPLOYED",
      "DEPLOYED, COMBINATION" = "DEPLOYED",
      "DEPLOYED, SIDE" = "DEPLOYED",   
      "DEPLOYED OTHER (KNEE, AIR, BELT, ETC.)" = "DEPLOYED"
      ),
  DRIVER_VISION = recode(DRIVER_VISION,
      "MOVING VEHICLES" = "VEHICLES",
      "PARKED VEHICLES" = "VEHICLES",
      "BLINDED - SUNLIGHT" = "BLINDED",
      "BLINDED - HEADLIGHTS" = "BLINDED",
      "IMPROPER LANE CHANGE" = "IMPROPER DRIVING",
      "WRONG WAY/SIDE" = "IMPROPER DRIVING",
      "IMPROPER TURN" = "IMPROPER DRIVING",
      "IMPROPER TURN" = "IMPROPER DRIVING"
      ),
  PEDPEDAL_ACTION_TYPE = recode(PEDPEDAL_ACTION,
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
    ),
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



## number of people involved in accident in each month by gender

new.df = df %>%
  filter(Sex != 'UNKNOWN') %>%
  count(Month, Sex)

new.df %>% ggplot(aes(x=Month, y=n, color = Sex, group = Sex)) +
  geom_line() + 
  geom_point() +
  labs(
    title = str_wrap('Number of people involved in accident in each month by gender', width = 40),
    y = 'Number of People'
  )

save.fig('num_people_each_month_by_gender')




## number of people involved in accident in each month by age range

new.df = df %>%
  filter(Age.range != 'UNKNOWN') %>%
  count(Month, Age.range)

new.df %>% ggplot(aes(x=Month, y=Age.range,fill = n)) +
  geom_tile() +
  labs(
    title = str_wrap('Number of people involved in accident in each month by age',width=40),
    y = 'Age',
    fill = 'Number of People'
  ) + 
  scale_fill_gradient2()

save.fig('num_people_each_month_by_age')



## num people by speed limit and injury type

new.df = df %>%
  filter(
    INJURY_CLASSIFICATION != 'UNKNOWN', PERSON_TYPE!='UNKONWN') %>%
  count(INJURY_CLASSIFICATION, Speed.limit.range, PERSON_TYPE) %>%
  mutate(
    n = log10(n),
    PERSON_TYPE = factor(PERSON_TYPE, levels = c("DRIVER", "PASSENGER", 'PEDESTRIAN', 'BICYCLE', 'NON-MOTOR VEHICLE', 'NON-CONTACT VEHICLE'))
  )

new.df %>% ggplot(aes(x=Speed.limit.range, y=n, fill = INJURY_CLASSIFICATION)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  facet_wrap(vars(PERSON_TYPE)) + 
  labs(
    title = str_wrap('Number of people involved in accident by speed limit and injury type', width=40),
    x = 'Speed Limit',
    y = 'Number of People (log base 10)',
    fill = 'Injyry Type'
  ) + 
  theme(legend.position = 'bottom') + 
  guides(fill = guide_legend(nrow=2))

save.fig('num_people_in_accident_by_speed_limit_and_injury_type')



## num people by work zone type, age, sex

new.df = df %>%
  filter(
    Age.range != 'UNKNOWN' & !Sex %in% c('UNKNOWN','Other')
  ) %>%
  count(WORK_ZONE_TYPE, Age.range, Sex)

new.df %>% ggplot(aes(x=Age.range, y=WORK_ZONE_TYPE, fill = n)) + 
  geom_tile() +
  facet_wrap(vars(Sex), nrow=2) + 
  labs(
    title = str_wrap('Number of people involved in accident by age and type of work zone',width=40),
    x = 'Age',
    y = 'Work Zone Type',
    fill = 'Number of People'
  ) +
  scale_fill_gradient2(low = "blue", mid = "yellow", high = "blue")

save.fig('num_people_in_accident_by_age_and_workzone_type')



## num people by lightning condition and pedpedal visibility

new.df = df %>%
  filter(LIGHTING_CONDITION != 'UNKNOWN' & PEDPEDAL_VISIBILITY != 'UNKNOWN') %>%
  count(LIGHTING_CONDITION, PEDPEDAL_VISIBILITY)

new.df %>% ggplot(aes(x=n, y=LIGHTING_CONDITION, fill = PEDPEDAL_VISIBILITY)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  labs(
    title = str_wrap('Number of poeple involved in accident by lightning condition and visibility', width=40),
    x = 'Number of People',
    y = 'Lightning Condition',
    fill = 'Pedpedal Visibility'
  ) + 
  scale_y_discrete(labels = label_wrap_gen(10))

save.fig('num_people_in_accident_by_light_cond_and_pedpedal_vis')



## relationship between driver's action and crash type

new.df = df %>%
  filter(
    PERSON_TYPE == 'DRIVER' & 
      DRIVER_ACTION != 'UNKNOWN') %>%
  mutate(
    DRIVER_ACTION = recode(DRIVER_ACTION,
      "IMPROPER BACKING" = "IMPROPER DRIVING",
      "TOO FAST FOR CONDITIONS" = "IMPROPER DRIVING",
      "FOLLOWED TOO CLOSELY" = "IMPROPER DRIVING",
      "IMPROPER PASSING" = "IMPROPER DRIVING",
      "IMPROPER TURN" = "IMPROPER DRIVING",
      "IMPROPER PARKING" = "IMPROPER DRIVING",
      "WRONG WAY/SIDE" = "IMPROPER DRIVING",
      "OVERCORRECTED" = "IMPROPER DRIVING",
      "TEXTING" = "USE CELL PHONE",
      "CELL PHONE USE OTHER THAN TEXTING" = "USE CELL PHONE"
    ),
    FIRST_CRASH_TYPE = recode(FIRST_CRASH_TYPE,
      "REAR TO FRONT" = "REAR",
      "REAR END" = "REAR",
      "REAR TO SIDE" = "REAR",
      "REAR TO REAR" = "REAR",
      "SIDESWIPE SAME DIRECTION" = "SIDESWIPE",
      "SIDESWIPE OPPOSITE DIRECTION" = "SIDESWIPE"
    )
  ) %>%
  select(FIRST_CRASH_TYPE, DRIVER_ACTION)
  # count(FIRST_CRASH_TYPE, DRIVER_ACTION)

new.df %>% ggplot() + 
  geom_mosaic(aes(x=product(DRIVER_ACTION,FIRST_CRASH_TYPE), fill=DRIVER_ACTION), offset = 0.03) + 
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
    title = 'Number of accident by crash type and driver action',
    x = 'Crash Type',
    y = 'Driver Action'
    # fill = 'Driver Action'
  )

save.fig('crash_type_vs_driver_action',scale = 1.5)

# new.df %>% ggplot(aes(x=FIRST_CRASH_TYPE, y=DRIVER_ACTION, fill = n)) + 
#   geom_tile() +
#   theme(axis.text.x = element_text(angle = 30, vjust=0.5))



## top-10 contribution cause by physical condition

top.10.cause = df %>%
  count(PRIM_CONTRIBUTORY_CAUSE) %>%
  arrange(desc(n)) %>%
  head(10) %>%
  select(PRIM_CONTRIBUTORY_CAUSE)

new.df = df %>%
  filter(
    PERSON_TYPE == 'DRIVER' &
      PHYSICAL_CONDITION != 'UNKNOWN' &
      PRIM_CONTRIBUTORY_CAUSE %in% top.10.cause$PRIM_CONTRIBUTORY_CAUSE
  ) %>%
  select(PHYSICAL_CONDITION, PRIM_CONTRIBUTORY_CAUSE) %>%
  make_long(PHYSICAL_CONDITION, PRIM_CONTRIBUTORY_CAUSE)

new.df %>% ggplot(aes(x = x, 
                      next_x = next_x, 
                      node = node, 
                      next_node = next_node, 
                      fill = factor(node), 
                      label = node)) + 
   geom_sankey(flow.alpha = 0.5, 
               node.color = "black",
               show.legend = FALSE, space = 10000) + 
   geom_sankey_label(size = 5, color = "black", fill= "white", hjust = -0.2, space = 10000) + 
  labs(
    title = 'Top-10 accident cause by physical condition of driver',
    x = ''
  ) + 
  theme(
    plot.title = element_text(size = 24),
    axis.text.x = element_text(size = 20),
    axis.text.y = element_blank()
    ) + 
  scale_x_discrete(labels = c(
    'PHYSICAL CONDITION',
    'PRIMARY CONTRIBUTORY CAUSE'))

save.fig('top_accident_cause_by_physical_condition', scale=2.5)



## FIRST_CRASH_TYPE vs BAC_RESULT.VALUE vs gender

new.df = df %>%
  filter(
    PERSON_TYPE == 'DRIVER' & 
      FIRST_CRASH_TYPE != 'UNKNOWN' & 
      BAC_RESULT.VALUE != 'UNKNOWN' &
      Sex %in% c('Female', 'Male')) %>%
  select(FIRST_CRASH_TYPE, BAC_RESULT.VALUE, Sex) %>%
  mutate(
    BAC_RESULT.VALUE = as.numeric(BAC_RESULT.VALUE)*100,
    FIRST_CRASH_TYPE = recode(FIRST_CRASH_TYPE,
      "REAR TO FRONT" = "REAR",
      "REAR END" = "REAR",
      "REAR TO SIDE" = "REAR",
      "REAR TO REAR" = "REAR",
      "SIDESWIPE SAME DIRECTION" = "SIDESWIPE",
      "SIDESWIPE OPPOSITE DIRECTION" = "SIDESWIPE"
    )
  )

new.df %>% ggplot(aes(x = BAC_RESULT.VALUE, y=FIRST_CRASH_TYPE, color=Sex)) + 
  geom_boxplot() + 
  theme(legend.position = 'bottom') + 
  labs(
    title = str_wrap('Relationship between crash type and blood alcohol concentration of driver', width=30),
    x = 'Blood Alcohol Concentration (%)',
    y = 'Crash Type'
  )

save.fig('crash_type_vs_blook_alc_concentration')
