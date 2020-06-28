
library(shiny)
library(tidyverse)
library(DataExplorer)
tidydat <-
    readRDS("C:/Users/Zainab/Desktop/bst692_group2_breastCA/data/analysis_city20200601(1).rds")
cities <- str_remove(unique(tidydat$city), " city")
cities <- cities[order(cities)]

# Define UI for application that draws a histogram
ui <- fluidPage(        # Application title
    titlePanel("Predict Breast Cancer"),
    
    navbarPage("Breast Cancer",
        tabPanel("EDA",
            verbatimTextOutput("EDA"),
            sidebarLayout(
                sidebarPanel(
                    selectInput("CityName",
                                choices = cities,
                                label = "Please Choose a City")
                ),
                
                # Show plots of the generated distribution
                mainPanel(
                    h1("Barplot of data summary"),
                    br(),
                    plotOutput("bar"),
                    
                    h1("Distribution of the continuous variables"),
                    br(),
                    plotOutput("hist"),
                    
                    h1("Skewed data on a log scale"),
                    br(),
                    plotOutput("skew"),
                    
                    h1("Correlation heat map"),
                    br(),
                    plotOutput("corr"),
                )
            )),
            tabPanel("Models", verbatimTextOutput("Models")),
            tabPanel("Map", tableOutput("Map"))
        )
    )
# Define server logic required to draw a histogram
server <- function(input, output) {
    tidydat_reactive <- reactive({
        tidydat %>%
            filter(str_remove(city, " city") == input$CityName)
    })
    
    output$bar <- renderPlot({
        plot_intro(tidydat_reactive())
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
    
}


# Run the application
shinyApp(ui = ui, server = server)
