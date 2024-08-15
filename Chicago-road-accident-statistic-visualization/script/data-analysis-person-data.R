library(tidyverse)
library(ggmosaic)

options(scipen=1000000)
theme_set(theme_bw())

df = read.csv('../dataset/cleaned/cleaned_person_data.csv')

fig.dir = '../figure/'

df = df %>%
  rename(Sex = SEX, Age = AGE) %>%
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
         "IMPROPER EQUIPMENT USED" = "SHOULD/LAP BELT USED IMPROPERLY",
         "IMPROPER EQUIPMENT USED" = "CHILD RESTRAINT USED IMPROPERLY",
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
    )
  )


save.fig = function(fig.name, scale=1){
  ggsave(paste0(fig.dir, fig.name,'.png'), scale=scale)
}



## age range vs gender

new.df = df %>% 
  filter(!Sex %in% c('UNKNOWN', 'Other') & Age.range != 'UNKNOWN') %>%
  count(Age.range, Sex)
  
new.df %>% ggplot(aes(x=Age.range, y=n, fill=Sex)) + 
  geom_bar(stat = 'identity', position = 'dodge') + 
  labs(
    title = 'Number of people that involved in accidents by age and gender',
    x = 'Age Range',
    y = 'Number of people'
  ) + 
  theme(legend.position = 'bottom')

save.fig('num_people_by_age_and_gender')



## PEDPEDAL_ACTION vs PEDPEDAL_LOCATION

new.df = df %>%
  filter(
    !PEDPEDAL_ACTION_TYPE %in% c('UNKNOWN', "UNKNOWN/NA") &
      !PEDPEDAL_LOCATION %in% c('UNKNOWN', "UNKNOWN/NA")
  ) %>%
  select(PEDPEDAL_ACTION_TYPE, PEDPEDAL_LOCATION)
  # count(PEDPEDAL_ACTION_TYPE, PEDPEDAL_LOCATION)

new.df %>% ggplot() + 
  geom_mosaic(aes(x=product(PEDPEDAL_ACTION_TYPE,PEDPEDAL_LOCATION), fill=PEDPEDAL_ACTION_TYPE), offset = 0.06) + 
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
    title = 'Pedpedal location and action',
    x = 'Pedpedal Location',
    y = 'Pedpedal Action'
    # fill = 'Driver Action'
  )

# new.df %>% ggplot(aes(x=PEDPEDAL_LOCATION, y=PEDPEDAL_ACTION_TYPE, fill=n)) + 
#   geom_tile() + 
#   theme(axis.text.x = element_text(angle=15, vjust = 0.5)) + 
#   labs(
#     title = 'Number of pedestraints that involved in accidents by action and location',
#     x = 'Pedpedal Location',
#     y = 'Pedpedal Action',
#     fill = ''
#   )

save.fig('num_pedpedal_by_action_and_location', scale = 1)



## INJURY_CLASSIFICATION vs AIRBAG_DEPLOYED

new.df = df %>%
  filter(INJURY_CLASSIFICATION != 'UNKNOWN' & 
           AIRBAG_DEPLOYED %in% c("NO DEPLOYMENT", "DEPLOYED") &
           PERSON_TYPE %in% c('DRIVER', 'PASSENGER')) %>%
  count(INJURY_CLASSIFICATION, AIRBAG_DEPLOYED) %>%
  mutate(n = log10(n))

new.df %>% ggplot(aes(x=INJURY_CLASSIFICATION, y=n, fill = AIRBAG_DEPLOYED)) + 
  geom_bar(stat = 'identity', position = 'dodge') +
  theme(axis.text.x = element_text(angle=15, vjust = 0.5)) + 
  labs(
    title = 'Number of people that involved in accidents by injury type and airbag deployment',
    x = 'Injury Type',
    y = 'Number of people (log base 10)',
    fill = 'Airbag Deployment'
  ) 
  # theme(legend.position = 'bottom')

save.fig('num_people_by_injury_and_airbag')



## INJURY_CLASSIFICATION vs SAFETY_EQUIPMENT

new.df = df %>%
  filter(INJURY_CLASSIFICATION != 'UNKNOWN' & 
           SAFETY_EQUIPMENT != 'UNKNOWN') %>%
  count(INJURY_CLASSIFICATION, SAFETY_EQUIPMENT)
  # mutate(n = log10(n))

new.df %>% ggplot(aes(x=n, y=SAFETY_EQUIPMENT, fill = INJURY_CLASSIFICATION)) + 
  geom_bar(stat = 'identity', position = 'fill') +
  theme(axis.text.x = element_text(angle=15, vjust = 0.5)) + 
  labs(
    title = 'Number of people that involved in accidents by injury type and safety equipment',
    x = 'Ratio',
    y = 'Safety Equipment',
    fill = 'Injury Type'
  )
  # theme(legend.position = 'bottom')

save.fig('num_people_by_injury_and_safety_equipment')



## INJURY_CLASSIFICATION vs ejection

new.df = df %>%
  filter(INJURY_CLASSIFICATION != 'UNKNOWN' & 
           !EJECTION %in% c('UNKNOWN', 'NONE')) %>%
  count(INJURY_CLASSIFICATION, EJECTION)
  # mutate(n = log10(n))

new.df %>% ggplot(aes(x=INJURY_CLASSIFICATION, y=n, fill = EJECTION)) + 
  geom_bar(stat = 'identity', position = 'dodge') +
  theme(axis.text.x = element_text(angle=15, vjust = 0.5)) + 
  labs(
    title = 'Number of people that involved in accidents by injury type and ejection status',
    x = 'Injury Type',
    y = 'Number of people',
    fill = 'Ejection Status'
  ) +
  theme(legend.position = 'bottom')

save.fig('num_people_by_injury_and_ejection')



## INJURY_CLASSIFICATION vs driver vision

new.df = df %>%
  filter(INJURY_CLASSIFICATION != 'UNKNOWN' & 
           DRIVER_VISION != 'UNKNOWN' & 
           PERSON_TYPE == 'DRIVER') %>%
  count(INJURY_CLASSIFICATION, DRIVER_VISION)
  # mutate(n = log10(n))

new.df %>% ggplot(aes(x=n, y=DRIVER_VISION, fill = INJURY_CLASSIFICATION)) + 
  geom_bar(stat = 'identity', position = 'fill') +
  theme(axis.text.x = element_text(angle=15, vjust = 0.5)) + 
  labs(
    title = 'Number of driver that involved in accidents by injury type and vision',
    x = 'Ratio',
    y = 'Vision Type',
    fill = 'Injury Type'
  )
  # theme(legend.position = 'bottom')

save.fig('num_driver_by_injury_and_vision')

