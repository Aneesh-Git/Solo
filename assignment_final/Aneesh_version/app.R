#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
# Find out more about building applications with Shiny here:
#    http://shiny.rstudio.com/

suppressPackageStartupMessages(library(tidyverse))
library(shiny)
library(janitor)
library(skimr)
library(DataExplorer)
library(leaflet)
library(sp)
library(sf)
library(rgdal)
library(tidycensus)
# loaded necessary datasets 
# bca_cleaned <- readRDS("./assignment_final/data/bca_cleaned.rds")
# income_insurance <- readRDS("./assignment_final/data/income_insurance.rds")

# created character vector of 25 places in Miami-Dade county from assignment 4
# cities <- bca_cleaned %>% select(city) %>% unique()

# created reactive function for bca_cleaned
# bca_react <- reactive({
   #  bca_cleaned %>% filter(city == input$cityname)
# })
# non-reactive version is just the "bca_cleaned" object

# created reactive and non-reactive versions of "no insurance" vector
# insur_react <- reactive({
#     str_remove(income_insurance$NAME, ", Florida") %>%
#         as_tibble() %>%
#         rename(city = value) %>%
#         filter(city == input$cityname)
# })


# Define UI for application that jpredicts breast cancer
ui <- fluidPage(
    
    # App title ----
    titlePanel("Predicting Late-Stage Breast Cancer"),
    
    # Sidebar layout with input and output definitions ----
    sidebarLayout(
        
        sidebarPanel(
            aboutThisApp <- renderText({ 
                "This is an interactive web application that looks at the 
                predictions of late-stage breast cancer in females within 
                Miami-Dade County. You can either choose to examine all females
                OR by the city you select. Have fun exploring!"
                }),
            aboutThisApp()
        ),
        
        # Main panel for displaying outputs ----
        mainPanel(
            
            # Output: Tabset w/ plot, summary, and table ----
            tabsetPanel(type = "tabs",
                        tabPanel(
                            "General Information",
                            selectInput(
                                "cityname", label = h4("Select a city:"), 
                                choices = cities, 
                                selected = "Pembroke Pines city"
                            ),
                            plotOutput("bar"),
                            plotOutput("hist"),
                            plotOutput("corr")
                            ),
                        tabPanel("Modeling"),
                        tabPanel("Insurance Map"),
                        tabPanel("Summary Information"),
                        tabPanel("Map",
                                 leafletOutput("myMap")
                                 )
                        )
                    )
                )
)

server <- function(input, output) {
    
    bca_react <- reactive({
        bca_cleaned %>% filter(city == input$cityname)
        })
    
    output$bar <- renderPlot({
        plot_intro(bca_react())
    })
    output$hist <- renderPlot({
        plot_histogram(bca_react(), ncol = 2)
    })
    output$corr <- renderPlot({
        plot_correlation(bca_react(), type = "continuous")
    })
    
    output$map <- renderLeaflet({ 
        tidycensus::census_api_key
        
        # get data from  census 
        uninsured <- get_acs(geography = "place",
                             variables = "S2701_C04_001",
                             state = 12,
                             year = 2018)
        # load counties data
        FL <- tigris::places(state = "FL")
        # saveRDS(uninsured,"uninsured.rds") save to local 
        saveRDS(FL,"FL.rds")
        FL <- readRDS("~/Documents/bst692/bst692_group2_breastCA/final project/FL.rds")
        FL_SF <- sf::st_as_sf(FL)
        
        uninsured_FL <- FL_SF %>% 
            left_join(uninsured, by="GEOID")
        
        pal <- colorBin("RdYlBu", domain = uninsured_FL$estimate)
        
        leaflet() %>% 
            addProviderTiles(providers$Esri.NatGeoWorldMap) %>% 
            addPolygons(data = uninsured_FL,
                        fillColor = ~pal(estimate),
                        color = "black",
                        popup = uninsured_FL$NAMELSAD) %>% 
            addLegend(position = "topright",
                      pal = pal,
                      values = uninsured_FL$estimate)
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
