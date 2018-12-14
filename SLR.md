Correlation Coefficient
================

Data:

``` r
pizza <- read.csv("F:/Statistics/6. Regression/pizza.csv")
```

Scatter Plot:

``` r
plot(pizza$Promote,pizza$Sales)
```

![](SLR_files/figure-markdown_github/unnamed-chunk-2-1.png)

Correlation Coefficient:

``` r
cor(pizza$Promote,pizza$Sales)
```

    ## [1] 0.9943917
