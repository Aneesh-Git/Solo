############################# Data Cleaning Setup #############################

suppressPackageStartupMessages(library(tidyverse))
library(C50)
library(randomForestSRC)
library(randomForest)
library(conflicted)
library(janitor)
library(recipes)
library(caret)
library(class)
options(scipen = 999)
data <- readRDS("./Aneesh_version/data/analysis_city20200601.rds")
tidydat <- readRDS("./data/tidydat.rds")
cities <- str_remove(unique(tidydat$city), " city")
cities <- cities[order(cities)]

data.clean <- data %>% 
  rename(Chemotherapy = 
    `Chemotherapy(0=none, 1=Neoadjuvant(before surgery) 2=adjuvant(after surgery), 3=only chemotherapy/no surgery, 4=recommended, unnkown if received)`) %>% 
  mutate(Chemotherapy = as.factor(Chemotherapy)) %>% 
  mutate(Chemotherapy = fct_recode(Chemotherapy,
                                 "None"= "0",
                                 "Neoadjuvant" = "1",
                                 "Adjuvant"= "2",
                                 "Only Chemotherapy & No Surgery"= "3",
                                 "Recommended"= "4"))


data.clean <- data.clean %>%
  rename(SurgType = "SurgType (0=none, 1=lumpectomy, 2=mastectomy/MRM, 3=unknown)") %>%
  mutate(SurgType = as_factor(SurgType)) %>%
  mutate(SurgType = fct_recode(SurgType,
                               "None" = "0",
                               "Lumpectomy" = "1",
                               "Mastectomy/MRM" = "2",
                               "Unknown" = "3"))


data.clean <- data.clean %>%
  mutate(Facility = fct_recode(Facility,
                               "Jackson" = "JMH",
                               "Sylvester" = "SYL"))


data.clean <- data.clean %>%
  mutate(tumorGrade = fct_recode(tumorGrade,
                                 "Anaplastic" = "anaplastic",
                                 "Poor/Unknown" = "poor/unknown",
                                 "Well/Moderate" = "well/mod"))

data.clean <- data.clean %>%
  mutate(finalClinicalStage = as_factor(finalClinicalStage)) %>%
  mutate(finalClinicalStage = fct_recode(finalClinicalStage, 
                                         "I" = "1",
                                         "II" = "2",
                                         "III" = "3",
                                         "IV" = "4",
                                         "Unknown" = "5"))


data.clean <- data.clean %>%
  mutate(
    finalPathStage = fct_recode(finalPathStage, 
                                     "I" = "1", 
                                     "II" = "2", 
                                     "III" = "3", 
                                     "IV" = "4",
                                     "DCIS" = "dcis",
                                     "No Surgery/Unknown/Missing" = 
                                  "no surgery/unknown/missing"))


data.clean <- data.clean %>%
  mutate(RaceEthnicity = factor(case_when(
    Hispanic == 1 ~"Hispanic",
    Hispanic == 0 & Race1Desc == "Black" ~ "Non-Hispanic Black",
    Hispanic == 0 & Race1Desc == "White" ~ "Non-Hispanic White ")))

### Recode to make discrete variables into Factor
data.clean <- data.clean %>% 
  mutate_at(.vars = vars(Hispanic, Insurance, Race1Desc, city), .funs = as_factor) %>%
  mutate(Hispanic = fct_recode(Hispanic, 
                               "Hispanic" = "1",
                               "Non-Hispanic" = "0")) %>%
  mutate(Insurance = fct_recode(Insurance, 
                                "Medicare" = "Medicare",
                                "Medicaid" = "Medicaid",
                                "Other" = "OTHER"))


data.clean <- data.clean %>% 
  mutate(finalPathStage = fct_drop(finalPathStage, only = "(Missing)" ))

data.complete <-na.omit(data.clean)

data.complete<- data.complete %>% 
  select(-Hispanic, -Race1Desc)

# cities <- unique(data.complete$city)
# data.complete<-as.data.frame(data.complete)
# cities<-as.data.frame(cities)

data.pred <- data.complete %>% 
  mutate(finalClinicalStage = factor(case_when(
    finalClinicalStage == "I" ~ "Early Stage",
    finalClinicalStage == "II" ~ "Early Stage",
    finalClinicalStage == "III" ~ "Late Stage",
    finalClinicalStage == "IV" ~ "Late Stage")))


data.pred.pre <- data.pred %>% 
  select(- finalPathStage, - tumorGrade, - SurgType, - Chemotherapy)

data.pred.pre <- na.omit(data.pred.pre)



set.seed(123)
trainingSet1 <- createDataPartition(data.pred.pre$finalClinicalStage, p = 0.8, list= FALSE)

data.train.pre <- data.pred.pre %>% slice(trainingSet1)
data.test.pre <- data.pred.pre %>% slice(-trainingSet1)

formattable::percent(prop.table(table(data.train.pre$finalClinicalStage)))
formattable::percent(prop.table(table(data.test.pre$finalClinicalStage)))

train.pre.dummy <-data.train.pre %>% 
  recipe(finalClinicalStage~ ., data=data.train.pre) %>%  
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep(training = data.train.pre) %>% 
  juice()

test.pre.dummy <-data.test.pre %>% 
  recipe(finalClinicalStage~ .,data=data.test.pre) %>%  
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep(training = data.test.pre) %>% 
  juice()

knn.test.pre<-test.pre.dummy
knn.train.pre<-train.pre.dummy

knn.test.pre$finalClinicalStage<-(as.numeric(as.factor(knn.test.pre$finalClinicalStage))-1)
knn.train.pre$finalClinicalStage<-(as.numeric(as.factor(knn.train.pre$finalClinicalStage))-1)

set.seed(123)
data.pred <- na.omit(data.pred)
trainingSet2 <- createDataPartition(data.pred$finalClinicalStage, p = 0.8, list= FALSE)


data.train.all <- data.pred %>% slice(trainingSet2)
data.test.all <- data.pred %>% slice(-trainingSet2)


formattable::percent(prop.table(table(data.train.all$finalClinicalStage)))
formattable::percent(prop.table(table(data.test.all$finalClinicalStage)))

train.all.dummy <- data.train.all %>% 
  recipe(finalClinicalStage ~ .) %>%  
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep(training = data.train.all) %>% 
  juice()

test.all.dummy <- data.test.all %>% 
  recipe(finalClinicalStage ~ .) %>%  
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  prep(training = data.test.all) %>% 
  juice()

knn.test.all <- test.all.dummy
knn.train.all <- train.all.dummy

knn.test.all$finalClinicalStage <- (as.numeric(as.factor(knn.test.all$finalClinicalStage))-1)
knn.train.all$finalClinicalStage <- (as.numeric(as.factor(knn.train.all$finalClinicalStage))-1)

# tidydat <- readRDS("~/Documents/bst692/bst692_group2_breastCA/final project/tidydat.rds")
cities <- str_remove(unique(data$city), " city")
cities <- cities[order(cities)]


################################################## DATA PREP ##################################################
####################################################################################################

library(shiny)
library(tidyverse)
library(dplyr)
library(table1)
library(recipes)
library(caret)
library(class)
library(descr)
library(randomForestSRC)
library(randomForest)
library(DataExplorer)
library(tidycensus)
library(leaflet)


set.seed(234)


######################################## pre- data for rf ##############################

data <- na.omit(analysis_city20200601)
data <- data[!(data$finalClinicalStage == 5),]
data <- as.data.frame(data)

data <- data %>%
  mutate(race=case_when(Hispanic==0&Race1Desc=="Black" ~"NHB", Hispanic==0&Race1Desc=="White" ~"NHW", Hispanic==1~"Hispanic" ))%>%
  rename(Chemotherapy='Chemotherapy(0=none, 1=Neoadjuvant(before surgery) 2=adjuvant(after surgery), 3=only chemotherapy/no surgery, 4=recommended, unnkown if received)')%>%
  rename(Surgery=`SurgType (0=none, 1=lumpectomy, 2=mastectomy/MRM, 3=unknown)`)%>%
  select(-Chemotherapy,-Surgery,-Hispanic,-tumorGrade,-Race1Desc,-finalPathStage)  

data$finalClinicalStage[data$finalClinicalStage<3]<-0 # 1 for late stage
data$finalClinicalStage[data$finalClinicalStage>2]<-1

data <- 
  data %>% 
  mutate(finalClinicalStage = 
           fct_recode(as.factor(finalClinicalStage),"Early"= "0", "Late" = "1"))


train <- createDataPartition(data$finalClinicalStage, p=0.8, list=FALSE)
train_data <- data %>% slice(train)
test_data <- data %>% slice(-train)

# create dummy variables
dummy_train <- 
  recipe( ~ ., data = train_data) %>% 
  step_dummy(Facility,Insurance,city,race) %>%
  prep(training = train_data) %>%
  bake(new_data=train_data)

dummy_test <-
  recipe( ~ ., data = test_data) %>% 
  step_dummy(Facility,Insurance,city,race) %>%
  prep(training = test_data) %>%
  bake(new_data=test_data)

dummy_train <- as.data.frame(dummy_train)
saveRDS(dummy_train,"trainData_preSurg.rds")
dummy_test <- as.data.frame(dummy_test)
saveRDS(dummy_test,"testData_preSurg.rds")

############################################## full data for rf ##################################################

data <- na.omit(analysis_city20200601)
data <- data[!(data$finalClinicalStage==5),]
data <- as.data.frame(data)

data <- data %>%
  mutate(race=case_when(Hispanic==0&Race1Desc=="Black" ~"NHB", Hispanic==0&Race1Desc=="White" ~"NHW", Hispanic==1~"Hispanic" ))%>%
  rename(Chemotherapy='Chemotherapy(0=none, 1=Neoadjuvant(before surgery) 2=adjuvant(after surgery), 3=only chemotherapy/no surgery, 4=recommended, unnkown if received)')%>%
  rename(Surgery=`SurgType (0=none, 1=lumpectomy, 2=mastectomy/MRM, 3=unknown)`)
# select(-Chemotherapy,-Surgery,-Hispanic,-tumorGrade,-Race1Desc,-finalPathStage)  

data$finalClinicalStage[data$finalClinicalStage < 3] <- 0 # 1 for late stage
data$finalClinicalStage[data$finalClinicalStage > 2] <- 1
 
data <- data %>%
  mutate(
    finalClinicalStage=fct_recode(
      as.factor(finalClinicalStage),"Early" = "0", "Late" = "1")
    )

train <- createDataPartition(data$finalClinicalStage, p = 0.8 ,list = FALSE)
train_data <- data %>% slice(train)
test_data <- data %>% slice(- train)

# create dummy variables
dummy_train <- 
  recipe( ~ ., data = train_data) %>% 
  step_dummy(Facility, Insurance, city, race) %>%
  prep(training = train_data) %>%
  bake(new_data = train_data)

dummy_test <-
  recipe( ~ ., data = test_data) %>% 
  step_dummy(Facility, Insurance, city, race) %>%
  prep(training = test_data) %>%
  bake(new_data = test_data)

dummy_train <- as.data.frame(dummy_train)
saveRDS(dummy_train,"trainData_allVar.rds")
dummy_test <- as.data.frame(dummy_test)
saveRDS(dummy_test,"testData_allVar.rds")
