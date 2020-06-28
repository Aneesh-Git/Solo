#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Old Faithful Geyser Data"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            column(4,
                   numericInput("x", "Value", 5),
                   br(),
                   actionButton("button", "Show")
            ),
              
        ),

        # Show a plot of the generated distribution
        mainPanel(
            column(8, tableOutput("table"))
           
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {

    # Take an action every time button is pressed; # here, we just print a message to the console
    observeEvent(input$button, {
        cat("Showing", input$x, "rows\n")
    })
    # Take a reactive dependency on input$button, but
    # not on any of the stuff inside the function
    df <- eventReactive(input$button, {
        head(cars, input$x)
    })
    output$table <- renderTable({
        df()
    })
    
}

# Run the application 
shinyApp(ui = ui, server = server)
