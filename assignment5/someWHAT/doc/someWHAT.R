## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(someWHAT)

## ----message=FALSE------------------------------------------------------------
x<-round(rnorm(3,0,1),1)
x
some(x)

## ----message=FALSE,error=TRUE-------------------------------------------------
x<-c("hello",1,2)
x
some(x)


## ----message=FALSE,error=TRUE-------------------------------------------------
x<-c(5,NA,1)
x
some(x)


