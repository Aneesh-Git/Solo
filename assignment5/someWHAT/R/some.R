#' Calculate the sum of vector elements
#'
#' @param x A vector that needs a sum
#'
#' @return The sum of vector elements
#' @export some
#'
#' @examples
#' some(rep(1,3))
#'
some<- function(x) {
  if (!is.numeric(x)){
    stop("Argument `x` should be numeric", call.=FALSE)
  }
  if(any(is.na(x))){
    stop("Missing values are not allowed", call.=TRUE)
  }
  sum(x)
}



