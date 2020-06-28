# Assignment 5

To test and check my package, in your local, switch to jiangnanBranch, go to "someWHAT" folder, click "someWHAT.Rproj"---> "More" -->"Check package"/"Test package". To view the vignette file, in your console, type the following code chunk: 

```{r}
devtools::install(build_vignettes = TRUE)
browseVignettes("someWHAT")
```
To install and view the help page of the package, click "Install and Restart", then search for `someWHAT` in Rstudio Packages panel

Or directly install the source or binary package:
`someWHAT_0.0.1.tar.gz` or `someWHAT_0.0.1.tgz`

