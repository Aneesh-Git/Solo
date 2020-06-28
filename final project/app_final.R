
# Load Data
load("./data/shiny_data.Rdata")
library(tidyverse)
library(shiny)
library(leaflet)
library(tidycensus)

# User Interface
ui <- fluidPage(  theme = shinythemes::shinytheme("cyborg"), # Application title
                  titlePanel(
                    h4("Disclaimer: This work is done on simulated data inspired by patients seen in Miami, FL", align = "Left", 
                       h4("Creators: Aneesh Chandramouli, Jiangnan Lyu, Zainab Alkhater"), align = "Left")
                  ),
                  
                  navbarPage("Florida Breast Cancer Prediction",
                             #tabsetPanel(
                             tabPanel("About this App",
                                      h3("Welcome to the exciting world of predicting breast cancer!"),
                                      br(),
                                      h4("The purpose of this web application is to make predictions about women 
                                          with breast cancer. More specifically, we want to examine a sample of 
                                          women in the state of Florida with breast cancer and use models to 
                                          correctly predict which women have late-stage breast cancer (compared to 
                                          women who have early-stage breast cancer). In our app, we use four models 
                                          that are based on four different machine-learning algorithms to determine 
                                          which predictors are associated with the occurrence late-stage breast cancer."),
                                      tags$div(),
                                      h4("( To learn more about machine-learning algorithms, click", 
                                         a("here", 
                                           href = "https://towardsdatascience.com/machine-learning-algorithms-in-laymans-terms-part-1-d0368d769a7b"), ")"),
                                      tags$div(),
                                      h4("Our fun-filled journey will begin with an exploration of our data, where we explore each variable in relation
                                         to whether or not a women has late-stage breast cancer. We will then look at a map of Florida and see the 
                                         how many Floridians do not have health insurance within a variety of locations. We will end our journey
                                         by examining the results of the models of our breast cancer predictions, as well as summarizing the key
                                         key points from our results."),
                                      tags$div(),
                                      h4("For your reference, we have included a section where we explain what each of our machine-learning 
                                         models are and the concepts behind them."),
                                      tags$div(),
                                      h4("Without further adieu, let's get started!"),
                                      br(),
                                      h4(actionLink("link", label="Click Here to Learn More about Breast Cancer", icon = icon("th"), 
                                                    onclick ="window.open('https://www.cdc.gov/cancer/breast/index.htm',
                                                 '_blank')")),
                                      br(),
                                      tags$div(img(src = "ML.jpg", width = 750, height = 400, align = "left"),
                                      img(src = "breastcancer.jpg", width = 600, height = 400, align = "right"))
                             ),
                             
                             tabPanel("Exploring the Data",
                                      sidebarLayout(
                                        sidebarPanel(
                                          selectInput("CityName",
                                                      choices = cities,
                                                      label = "Please Choose a City"),
                                          actionButton("go","Submit")
                                        ),

                                        # Show plots of the generated distribution
                                        mainPanel(
                                          h2("An Overview of the Dataset"),
                                          br(),
                                          
                                          h4("Table"),
                                          br(),
                                          tableOutput("t"), ########## ADD TABLE
                                          
                                          h4("Principal Components of Data Summary"),
                                          br(),
                                          plotOutput("prcomp"),
                                          
                                          h4("Distribution of Continuous/Discrete Variables: Age, Median Income, Final Clinical Stage"),
                                          br(),
                                          plotOutput("hist"),
                                          
                                          h4("Correlation Heatmap"),
                                          br(),
                                          plotOutput("corr")
                                        )
                                      )), 
                             
                             ############## Map tab #########################################
                             tabPanel("Map",
                                      h3("Visualizing the Uninsured Population in Florida"),
                                      leafletOutput("map"),
                                      br(),
                                      br(),
                                      actionLink("link", label="Learn More", icon = icon("th"), onclick = "window.open('https://www.census.gov/data.html')")
                             ),

                             tabPanel("Model Predictions",
                                      selectInput("select", label = "Predictors",
                                                  choices = c("Pre-surgery", "All")),
                                      h5("NOTE: The pre-surgery variables are Hispanic, Age, Median Income, Race, Facility, Insurance, and City"),            
                                      navlistPanel(
                                        
                                        tabPanel("k-Nearest Neighbors",
                                                 h2("Statistical Analysis Output:"),
                                                 br(),
                                                 verbatimTextOutput("knn"), 
                                                 br(), 
                                                 h2("Interpretation and Comments:"),
                                                 br(),
                                                 textOutput("c1"),
                                                 br(),
                                                 actionLink("klink", label = "Learn More about Kappa Interpreation", icon = icon("th"),onclick ="window.open('https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f')"),
                                                 br()),
                                        
                                        tabPanel("Logistic Regression",
                                                 h2("Statistical Analysis Output:"),
                                                 br(),
                                                 verbatimTextOutput("log"),
                                                 br(), 
                                                 h2("Interpretation and Comments:"),
                                                 br(),
                                                 textOutput("c2"),
                                                 br(),
                                                 actionLink("klink", label="Learn More about Logistic Regression", icon = icon("th"),onclick ="window.open('https://www.machinelearningplus.com/machine-learning/logistic-regression-tutorial-examples-r/')"),
                                                 br()),
                                        
                                        tabPanel("Classification Tree",
                                                 h2("Statistical Analysis Output:"),
                                                 br(),
                                                 verbatimTextOutput("tree"),
                                                 br(), 
                                                 h2("Interpretation and Comments:"),
                                                 br(),
                                                 textOutput("c3"),
                                                 br(),
                                                 actionLink("klink", label="Learn More about Classification Tree", icon = icon("th"), onclick ="window.open(
                                                 'https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm')"),
                                                 br()),
                                        
                                        tabPanel("Random Forest",
                                                 h2("Statistical Analysis Output:"),
                                                 br(),
                                                 verbatimTextOutput("rf"),
                                                 br(), 
                                                 h2("Interpretation and Comments:"),
                                                 br(),
                                                 textOutput("c4"),
                                                 br(),
                                                 actionLink("klink", label = "Learn More about Random Forest", icon = icon("th"), onclick = "window.open(
                                                 'https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/')"),
                                                 br()),
                                        
                                          tabPanel("Variable Importance",
                                                   br(),
                                                   h4(
                                                     "Because our random forest algorithm has the , we are displaying two plots of the relative
                                                     importance (from highest to lowest) of the variables. These two plots show indicate how important 
                                                     that variable is in classifying the data. The plot shows each variable on the y-axis, and their 
                                                     importance on the x-axis. They are ordered top-to-bottom as most- to least-important. Therefore, 
                                                     the most important variables are at the top and an estimate of their importance is given by the 
                                                     position of the dot on the x-axis."
                                                   ),
                                                   tags$div(),
                                                   h4("The criteria we are using for importance on the x-axis is the mean decrease in Gini, which 
                                                       indicates the average decrease in impurity associated with a given variable. The higher the 
                                                       mean decrease in Gini, the more important a variable is in our random forest model."),
                                                   tags$div(),
                                                   h4("In both plots, we can see that age and median income are the two most important variables
                                                     in our model compared to the rest. Both plots show a noticeable drop-off in the importance 
                                                     (across the x-axis) after age and median income."),
                                                   h2("Pre-Surgery Variables Only:"),
                                                   br(),
                                                   tags$div(img(src = "Rplot_PRE_TEST.png", width = 600, height = 650)),
                                                   h2("All Variables:"),
                                                   br(),
                                                   tags$div(img(src = "Rplot_ALL_TEST.png", width = 600, height = 650))
                                            
                                          ),
                                        
                                          tabPanel("Overall Statements",
                                                   h2("Overall Statements:"),
                                                   br(),
                                                   h3("Predictions with Models Involving Only Pre-Surgery Variables:"),
                                                   br(),
                                                   h4("If we were to just look at the p-value for all of our models
                                                      involving only pre-surgery variables, we may erroneously conclude
                                                      that our models are significant. While our models may be statistically
                                                      significant, they have virtually no predictive power due to the extremely
                                                      low Kappa values. In other words, we cannot state that our pre-surgery
                                                      predictors are clinically or practically significant, and moreover, our 
                                                      models in this case are useless. We essentially have no ability to predict
                                                      late-stage breast cancer using only pre-surgery variables with these models."),
                                                   br(),
                                                   h3(" Predictions with Models Involving All Variables:"),
                                                   br(),
                                                   h4("Again, if we were to just look at the p-value for all of our models
                                                      involving all variables, we may erroneously conclude
                                                      that our models are significant. In this case, each of the models when 
                                                      considering all variables have Kappa values that are moderately to very 
                                                      reliable. Because our Kappa values of these models containing all variables 
                                                      are notably higher than the models containing only pre-surgery variables,
                                                      our models in this scenario have a much better ability to predict late-stage
                                                      breast cancer. The classification tree and random forest models in particular
                                                      have good Kappa values, meaning that these models, when considering all variables,
                                                      have the greatest ability to predcit late-stage breast cancer."
                                                      ),
                                                   br()
                                                   )

                                      ), 
                          ),      
                             
                          ############## Explanation for the models tab #########################################
                             
                             tabPanel("Model Explanations",
                                      h2("The algorithms we used:"),
                                      br(),
                                      h2("1) k-Nearest Neighbors:"),
                                      h4("Our aim of using k-nearest neighbors (KNN) is to check how closely
                                      the properties of the new patient of breast cancer are is related to any
                                      one of the already known categories of stage breast cancer. In this model,
                                      the nearest neighbor which we want to check its characteristics will be defined
                                      by value “k”. By using the knn() function within the R package known as caret(),
                                      we determine the optimal number of neighbors to classify a given set of cases. 
                                      If the majority of neighbors belong to a certain category from breast cancer stage 
                                      that are late or not within those five nearest neighbors, then we will be chosen as
                                      the kind of upcoming patients."),
                                      br(),
                                      h3("Strengths and Weaknesses:"),
                                      h4("The knn algorithm is more appropriate if the model contains the numeric variables. However, this does not 
                                         mean that it cannot work with categorical variables, but in case we will have a mix model of both categorical 
                                         and numeric predictors, like our model, then it demands other approach. But if all predictors are numeric, then knn 
                                         is best because we are dealing with the distance and for that we need hard numbers. Also, when we split our data into 
                                         training and testing sets, the data should have already be normalized, that means if we have different variables have 
                                         different scaling units, like age and median income in this model, we will need make them have the same value that 
                                         is between 0 and 1."),
                                      actionLink("link", label="Learn More about k-Nearest Neighbors", icon = icon("th"), onclick ="window.open('https://towardsdatascience.com/interpretation-of-kappa-values-2acd1ca7b18f', '_blank')"),
                                      br(),
                                      h2("2) Logistic Regression:"),
                                      h4("Binary logistic regression is a predictive modeling algorithm that is used when the outcome
                                        variable is binary. That is, it can take only two values like 1 or
                                        0 which are that in our model 1 for late stage of breast cancer and 0 for else
                                        stage . The aim here is to determine a mathematical equation that can be used to
                                        predict the probability of late clinical stage of breast cancer for each patient.
                                        Once the equation is established, it can be used to predict the clinical stage of
                                        breast cancer."),
                                      br(),
                                      h3("Strengths and Weaknesses:"),
                                      h4("The logistic regression is very efficient, does not require too many computational resources,
                                      highly interpretable, it doesn’t require input features to be scaled, it doesn’t require any tuning, it’s easy to regularize, 
                                      and it outputs well-calibrated predicted probabilities. However, the problem of logistic regression is that when you create a logistic
                                      model with factor variables as features, it converts each level in the factor into a dummy binary variable of 1's and 0's. 
                                      For instance, cell shape is a factor with 5 levels. When you use logistic regression model class as a function of cell shape, 
                                        the cell shape will be split into 4 different binary categorical variables before building the model."),                                      
                                      actionLink("link3", label="Learn More about Logistic Regression", icon = icon("th"), 
                                                 onclick ="window.open('https://www.machinelearningplus.com/machine-learning/logistic-regression-tutorial-examples-r/', '_blank')"),
                                      br(),
                                      h2("3) Classification Tree:"),
                                      h4("When using the classification tree algorithm, we started out with our 
                                        entire dataset and divided it into several subsets of data. Every divide, 
                                        or split, was made based on a binary variable, where the only reponses were 
                                        'Yes' or 'No'. The very first split was made using the binary variable that 
                                        best separated women with late-stage breast cancer and women who were not 
                                        in the late stage. We then used the next best binary variable (in terms of 
                                        separating the late and non-late stage cases) to do our second split. We 
                                        continued splitting the data in this exact manner until nearly all of the 
                                        individuals within a subgroup were part of the SAME CLASS. In other words, 
                                        we kept dividing our data into subests until we had samples where almost 
                                        all women were either late stage or non-late stage (also known as a nearly 
                                        'pure' or 'homogenous' samples)."),
                                      br(),
                                      h3("Strengths and Weaknesses:"),
                                      h4(" The classification tree algorithm is easy to use, follow, interpret, and explain.
                                        However, the question often arises as to how much to split the data, and this
                                        becomes problematic as there is often no simple answer. Split too much and 
                                        the data will be overfitted; the model would fit on noise, and predictions
                                        are thrown off. Split too little and the data would be underfitted; the model 
                                        won't capture enough of the data’s characteristics and wouldn't be able to 
                                        distinguish between the two classes."), 
                                      actionLink("link4", label = "Learn More about Classification Trees", icon = icon("th"), onclick = "window.open('https://towardsdatascience.com/decision-tree-classification-de64fc4d5aac', '_blank')"),
                                      br(),
                                      h2("4) Random Forest:"),
                                      h4("The random forest algorithm is a model made up of many classification trees.
                                        Each of the trees is created from a random sample of data (with replacement),
                                        At each binary split of the data, we randomly select a subset of variables that can 
                                        potentially be selected to be split on, rather than evaluate all of the predictor 
                                        variables. Every time we randomly select some variables, we choose the best 
                                        feature/variable out of this random subset and keep splitting until we have a tree 
                                        with the most 'pure' samples possible. We do this over and over and until we produce 
                                        many trees. Once we have lots of trees, we then merge all of them together. This 
                                        prevents overfitting of the data and allows us to determine a stable predictor."),
                                       br(),
                                       h3("Strengths and Weaknesses:"),
                                       h4("The main strength of random forests is that they mitigate the issue of overfitting
                                         because they are an aggregation of many individual classification trees. By training 
                                         many (think hundreds) of trees, overall instability is reduced and the important 
                                         variables contributing to the model can be clearly assessed. Random forets can also
                                         deal with large amounts of data as well as missing data very well. However, they
                                         are often time-consuming to build, explain (i.e. not very intuitive), and interpet."),
                                       actionLink("link5", label = "Learn More about Random Forests", icon = icon("th"), onclick = "window.open('https://towardsdatascience.com/understanding-random-forest-58381e0602d2', '_blank')")
                                    )
                             )
                      )



# Server

server <- function(input, output) {
  
  tidydat_reactive <- eventReactive(input$go,{
      na.omit(tidydat) %>%
      filter(str_remove(city, " city") == input$CityName)
  })
  observe(tidydat_reactive())
  
  
  
  ######################### data split for kNN and calssification tree #########################
  knn_split<- reactive({
    if (input$select=="Pre-surgery"){
      TRAIN<-knn.train.pre
      TEST<-knn.test.pre
      return(list(TRAIN,TEST))
    }
    if(input$select=="All"){ 
      TRAIN<-knn.train.all
      TEST<-knn.test.all
      return(list(TRAIN,TEST))
    }})
  
  ######################### data split for random forest and logistic  #########################
  rf_split<- reactive({
    if (input$select=="Pre-surgery"){
      TRAIN<-train.pre.dummy
      TEST<-test.pre.dummy
      return(list(TRAIN,TEST))
    }
    if(input$select=="All"){ 
      TRAIN<-train.all.dummy
      TEST<-test.all.dummy
      return(list(TRAIN,TEST))
    }})
  
  
  
  ######################### return a confusion matrix output: knn #########################
  knn_pre<-reactive({
    set.seed(123)
    TRAIN<-knn_split()[[1]]
    TEST<-knn_split()[[2]]
    train_labels <- TRAIN %>% select(finalClinicalStage) %>% unlist()
    test_labels <- TEST %>%  select(finalClinicalStage) %>% unlist()
    test_pred <- knn(train = TRAIN,
                     test = TEST,
                     cl = train_labels,
                     k = 19)
    test_pred<-as.factor(if_else(as.factor(test_pred)=="0","Early Stage", "Late Stage"))
    test_labels<-as.factor(if_else(as.factor(test_labels)=="0","Early Stage", "Late Stage"))
    confusionMatrix(as.factor(test_pred),as.factor(test_labels),positive = "Late Stage")
  })
  
  
  ######################### return a confusion matrix output: logistic #########################
  log_pre<-reactive({
    set.seed(123)
    TRAIN<-rf_split()[[1]]
    TEST<-rf_split()[[2]]
    glm.fit <- glm(finalClinicalStage~.,TRAIN,family = "binomial")
    #summary(glm.fit)
    glm.prediction <- predict(glm.fit, TEST, type = "response")
    predicted <- as.factor(if_else(as.factor(if_else(glm.prediction <= 0.85, 0, 1))=="0","Early Stage", "Late Stage"))
    truth <- as.factor(TEST$finalClinicalStage)
    confusionMatrix(predicted, truth, positive = "Late Stage")
  })
  
  ######################### return a confusion matrix output: Classification tree #########################
  tree_pre<-reactive({
    set.seed(123)
    TRAIN<-rf_split()[[1]]
    TEST<-rf_split()[[2]]
    tree<-C5.0(TRAIN[-3],TRAIN$finalClinicalStage)
    predicted <- predict(tree, TEST)
    truth <- as.factor(TEST$finalClinicalStage)
    confusionMatrix(predicted, truth, positive = "Late Stage")
  })
  
  ######################### return a confusion matrix output:random forest  #########################
  rf_pre<- reactive({
    TRAIN<-rf_split()[[1]]
    TEST<-rf_split()[[2]]
    set.seed(123)
    m1<-randomForest(as.factor(finalClinicalStage)~.,data=TRAIN,ntree =100,importance = T) 
    pred1<-predict(m1,TEST) 
    confusionMatrix(pred1,as.factor(TEST$finalClinicalStage),positive = "Late Stage")
  }
  )
  
  output$knn<-renderPrint({knn_pre()})
  output$log<-renderPrint({log_pre()})
  output$tree<-renderPrint({tree_pre()})
  output$rf<-renderPrint({rf_pre()})
  
  
  
  ############ comments for outputs ########################################
  output$c1 <- renderText({
    if (input$select == "Pre-surgery"){
      print(
        "The accuracy of the model containing all variables is approximately 88.4%
            which is 8 percentage points above the baseline accuracy of 80% which is not really
            that impressive,
            because model accuracy is the measurement used to determine which model is best
            at identifying relationships and patterns between variables in a dataset based
            on the input, or training, data. The Kappa shows 34%, which is poor.
            The kappa means frequently used to test interrater reliability.
            The importance of rater reliability lies in the fact that it represents
            the extent to which the data collected in the study are correct representations
            of the variables measured."
      )
    } else if (input$select == "All") {
      print(
        "The accuracy of the model containing all variables is approximately 90%
            which is 10 percentage points above the baseline accuracy of 80% which is not really
            that impressive, because model accuracy is the measurement used to determine which
            model is best at identifying relationships and patterns between variables in a
            dataset based on the input, or training, data. The Kappa shows 51%, which is good.
            The kappa means frequently used to test interrater reliability. The importance of
            rater reliability lies in the fact that it represents the extent to which the data
            collected in the study are correct representations of the variables measured."
      )
    }
  })
  
  output$c2 <- renderText({
    if (input$select == "Pre-surgery"){
      print(
        "The accuracy of the model containing all variables is approximately
            84.7% which is 4 percentage points above the baseline accuracy of 80% which 
            is good, because model accuracy is the measurement used to determine which model 
            is best at identifying relationships and patterns between variables in a dataset 
            based on the input, or training, data. The Kappa shows 0, which is poor. 
            The kappa means frequently used to test interrater reliability. The importance 
            of rater reliability lies in the fact that it represents the extent to which the 
            data collected in the study are correct representations of the variables measured."
      )
    } else if (input$select == "All") {
      print(
        "The accuracy of the model containing all variables is approximately 92.2%
            which is 12 percentage points above the baseline accuracy of 80% which is not really
            that impressive, because model accuracy is the measurement used to determine which
            model is best at identifying relationships and patterns between variables in a
            dataset based on the input, or training, data. The Kappa shows 63.8, which is good.
            The kappa means frequently used to test interrater reliability. The importance of
            rater reliability lies in the fact that it represents the extent to which the data
            collected in the study are correct representations of the variables measured."
        
      )
    }
  })
  output$c3 <- renderText({
    if (input$select == "Pre-surgery"){
      print(
        "The results of our predictions using the classification tree model FOR 
    PRE-SURGERY VARIABLES ONLY are shown above. This image displays the results  
    of our predicted cases, labed as 'Predicted', versus actual cases, labeled 
    as 'Reference'. Our classification tree model accurately predicted 395 out 
    of the 466 total cases, leading to an accuracy of 84.76%. Another important 
    observation is that our Kappa value, shown a few rows below our Accuracy 
    value, is 0%. This means that our model is not any better at correctly 
    classifying the cases when random chance is accounted for. In other words, 
    we could have randomly guessed the correct classifcation of cases and gotten 
    the same result, so this particular model is useless!"
      )
    } else if (input$select == "All") {
      print(
        "The results of our predictions using the classification tree model FOR
    ALL VARIABLES are shown above. This image displays the results of our 
    predicted cases, labed as 'Predicted', versus actual cases, labeled as 
    'Reference'. Our classification tree model accurately predicted 441 out of 
    the 466 total cases, leading to an accuracy of 94.64%. Another important 
    observation is that our Kappa value, shown a few rows below our Accuracy 
    value, is 79.11%. This is a fairly reliable Kappa value and means that our 
    model is fairly good at correctly predicting breast cancer cases even when 
    random chance is taken into account. In other words, this particular model 
    is useful because it has significantly greater predictive power compared to 
    if we were to just randomly predict the cases!"
      )
    }
  })
  
  output$c4 <- renderText({
    if (input$select == "Pre-surgery") {
      print(
        "The results of our predictions using the random forest model FOR 
      PRE-SURGERY VARIABLES ONLY are shown above. This image displays the results  
      of our predicted cases, labed as 'Predicted', versus actual cases, labeled 
      as 'Reference'. Our classification tree model accurately predicted 395 out 
      of the 466 total cases, leading to an accuracy of 84.76%. Another important 
      observation is that our Kappa value, shown a few rows below our Accuracy 
      value, is only 8.92%. This means that our model is only 8.92% better at 
      correctly classifying the cases when random chance is accounted for. In 
      other words, this specific algorithm barely did a better job of making 
      correctly predicting breast cancer cases compared to if we were to 
      randomly guess, so this particular model is not useful!"
      )
    } else if (input$select == "All") {
      print(
        "The results of our predictions using the classification tree model FOR
      ALL VARIABLES are shown above. This image displays the results of our 
      predicted cases, labed as 'Predicted', versus actual cases, labeled as 
      'Reference'. Our classification tree model accurately predicted 449 out of 
      the 466 total cases, leading to an accuracy of 96.35%. Another important 
      observation is that our Kappa value, shown a few rows below our Accuracy 
      value, is 85.63%. This is a reliable Kappa value and means that our model
      is very good at correctly predicting breast cancer cases even when 
      random chance is taken into account. In other words, this particular model 
      is useful because it has significantly greater predictive power compared to 
      if we were to just randomly predict the cases!"
      )
    }
  })
  
  ######################### plots for the EDA tab   #########################
  output$t<-renderTable({
    tidydat_reactive() %>%
      summarise(`Median Income`=median(MedianIncome),
                `Mean Age` = mean(Age),
                `Hispanic Proportion`=sum(Hispanic)/length(Hispanic),
                `Late Stage Ratio`=sum(finalClinicalStage>=3)/length(finalClinicalStage),
                `N=Sample Size`=length(Hispanic)
                
      )
  })
  
  output$prcomp <- renderPlot({
    plot_prcomp(tidydat_reactive())
  })
  
  
  output$hist <- renderPlot({
    plot_histogram(tidydat_reactive(), ncol = 2)
  })
  
  output$skew <- renderPlot({
    plot_histogram(tidydat_reactive(),  ncol = 2, scale_x = "log10")
  })
  
  output$corr <- renderPlot({
    plot_correlation(tidydat_reactive(), type = "continuous")
  })
  
  
  ###################### Mapping ###################
  output$map <- renderLeaflet({ 
    
    tidycensus::census_api_key
    
    ############# get data from census #####################
    uninsured <- get_acs(geography = "place",
                         variables = "S2701_C04_001",
                         state = 12,
                         year = 2018)
    #saveRDS(uninsured,"uninsured.rds") save to local 
    
    #######load counties data ############
    FL=tigris::places(state = "FL")
    #saveRDS(FL,"FL.rds")
    #FL<-readRDS("~/Documents/bst692/bst692_group2_breastCA/final project/FL.rds")
    FL_SF=sf::st_as_sf(FL)
    
    uninsured_FL=FL_SF %>% 
      left_join(uninsured,by="GEOID")
    
    pal=colorBin("RdYlBu",domain = uninsured_FL$estimate)
    
    leaflet() %>% 
      addProviderTiles(providers$Esri.NatGeoWorldMap) %>% 
      addPolygons(data=uninsured_FL,
                  fillColor = ~pal(estimate),
                  color = "black",
                  popup=  uninsured_FL$NAMELSAD) %>% 
      addLegend(position = "topright",
                pal=pal,
                values= uninsured_FL$estimate)
  })
  
}


# Run the application
shinyApp(ui = ui, server = server)
