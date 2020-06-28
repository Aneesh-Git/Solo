test_that("some works", {
  expect_equal(some(c(-1,2,3)), 4)
})


test_that("Stops characters", {
  expect_error(some(c("hello","hey")), "Argument `x` should be numeric")
})


test_that("Stops missings", {
  expect_error(some(c(1,NA,0)), "Missing values are not allowed")
})
