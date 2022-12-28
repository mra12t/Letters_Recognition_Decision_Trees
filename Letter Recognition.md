Letter Recognition
================
2022-12-20

### About the Study and the Data Set

The letters_ABPR.csv file contains data for 3116 images of the letters
A, B, P, and R, which were produced by distorting 20 different fonts.
Each image is represented as a collection of pixels that are either “on”
or “off”, and the data includes statistics about the images as well as
the corresponding letter for each image. This data was obtained from the
UCI Machine Learning Repository.  
The dataset described contains 17 variables, including the letter
represented in the image (A, B, P, or R), the position and size of the
smallest bounding box enclosing the letter, various statistical measures
of the distribution of “on” pixels in the image (e.g. mean position,
mean squared position, mean of product of position and squared
position), and measures of the number of edges in the image when scanned
horizontally or vertically.

### EDA

``` r
lettersd = read.csv("letters_ABPR.csv")
str(lettersd)
```

    ## 'data.frame':    3116 obs. of  17 variables:
    ##  $ letter   : chr  "B" "A" "R" "B" ...
    ##  $ xbox     : int  4 1 5 5 3 8 2 3 8 6 ...
    ##  $ ybox     : int  2 1 9 9 6 10 6 7 14 10 ...
    ##  $ width    : int  5 3 5 7 4 8 4 5 7 8 ...
    ##  $ height   : int  4 2 7 7 4 6 4 5 8 8 ...
    ##  $ onpix    : int  4 1 6 10 2 6 3 3 4 7 ...
    ##  $ xbar     : int  8 8 6 9 4 7 6 12 5 8 ...
    ##  $ ybar     : int  7 2 11 8 14 7 7 2 10 5 ...
    ##  $ x2bar    : int  6 2 7 4 8 3 5 3 6 7 ...
    ##  $ y2bar    : int  6 2 3 4 1 5 5 2 3 5 ...
    ##  $ xybar    : int  7 8 7 6 11 8 6 10 12 7 ...
    ##  $ x2ybar   : int  6 2 3 8 6 4 5 2 5 6 ...
    ##  $ xy2bar   : int  6 8 9 6 3 8 7 9 4 6 ...
    ##  $ xedge    : int  2 1 2 6 0 6 3 2 4 3 ...
    ##  $ xedgeycor: int  8 6 7 11 10 6 7 6 10 9 ...
    ##  $ yedge    : int  7 2 5 8 4 7 5 3 4 8 ...
    ##  $ yedgexcor: int  10 7 11 7 8 7 8 8 8 9 ...

Let’s start by trying to predict whether a letter is B or not B and then
we can move on to predecting between the different letters.

So we will add a variable that gives True when the letter is B to the
dataset

``` r
lettersd$isB = as.factor(lettersd$letter == "B")
```

Now, let’s split the data set and establish a base line model. we will
set the seeds to an arbitrary number so that the results are
reproducable.

``` r
library(caTools)
set.seed(123)
spl = sample.split(lettersd$isB, SplitRatio = 0.5)
train = subset(lettersd, spl == TRUE)
test = subset(lettersd, spl == FALSE)
```

``` r
library(knitr)
kable(table(test$isB))
```

| Var1  | Freq |
|:------|-----:|
| FALSE | 1175 |
| TRUE  |  383 |

We can see that a base line model that predict the most likely outcome
on the Test set (Not B), will have an accuracy of
$(1175)/(1175 + 383)= 0.75$.

### Building a Classification Tree

``` r
library(rpart)
library(rpart.plot)
CARTb = rpart(isB ~ . -letter, data = train, method = "class")
prp(CARTb)
```

![](Letter%20Recognition_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

So the tree is quite a complicated one, let’s see the numerical values.

``` r
predb = predict(CARTb, newdata = test, type = "class")
kable(table(test$isB, predb))
```

|       | FALSE | TRUE |
|:------|------:|-----:|
| FALSE |  1143 |   32 |
| TRUE  |    55 |  328 |

Let’s see the accuracy of the model.

``` r
x = table(test$isB, predb)
(x[1] + x[4])/ sum(x)
```

    ## [1] 0.9441592

The accuracy is 94%.

### Building a Random Forest

``` r
library(randomForest)
```

    ## randomForest 4.7-1.1

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
set.seed(123)
RFb = randomForest(isB ~ .-letter, data = train)
importance(RFb)
```

    ##           MeanDecreaseGini
    ## xbox             11.338852
    ## ybox             13.421200
    ## width            11.687913
    ## height            9.736475
    ## onpix            13.548555
    ## xbar             16.276709
    ## ybar             41.646473
    ## x2bar            22.891303
    ## y2bar            77.084647
    ## xybar            28.311761
    ## x2ybar           31.012889
    ## xy2bar           47.671706
    ## xedge            35.639929
    ## xedgeycor        65.773045
    ## yedge           102.042352
    ## yedgexcor        48.234849

``` r
predRFb = predict(RFb, newdata = test)
kable(table(test$isB, predRFb))
```

|       | FALSE | TRUE |
|:------|------:|-----:|
| FALSE |  1165 |   10 |
| TRUE  |    18 |  365 |

Let’s calculate the accuracy of the random forest.

``` r
x = table(test$isB, predRFb)
(x[1]+x[4])/sum(x)
```

    ## [1] 0.9820282

The increase of the accuracy of the random forest model compared to the
CART model seems significant. The down side , however, is the
interpretability of the model.

Now Let’s build a model to predict what is the letter.

### Predicting A, B, R, and B.

``` r
lettersd$letter = as.factor(lettersd$letter)
set.seed(123)
spl = sample.split(lettersd$letter, SplitRatio = 0.5)
train1 = subset(lettersd, spl == TRUE)
test1 = subset(lettersd, spl == FALSE)
```

Let’s see the accuracy of a base line model.

``` r
x = table(test1$letter)
kable(x)
```

| Var1 | Freq |
|:-----|-----:|
| A    |  395 |
| B    |  383 |
| P    |  401 |
| R    |  379 |

A base line model on the test set would predict the most occuring
variable all the time, in this case it is P and the accuracy is.

``` r
401/nrow(test1)
```

    ## [1] 0.2573813

25.7%.

### Building a Classification Tree

``` r
set.seed(123)
CARTmodel = rpart(letter ~ . -isB, data = train1, method = "class")
prp(CARTmodel)
```

![](Letter%20Recognition_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

We can see that the tree is quite interpretable.

``` r
predl = predict(CARTmodel, newdata = test1, type = "class")
x = table(test1$letter, predl)
kable(x)
```

|     |   A |   B |   P |   R |
|:----|----:|----:|----:|----:|
| A   | 361 |   2 |   0 |  32 |
| B   |  14 | 293 |  13 |  63 |
| P   |   4 |  12 | 380 |   5 |
| R   |  10 |  16 |   7 | 346 |

``` r
#Accuracy
sum(diag(x))/ sum(x)
```

    ## [1] 0.885751

We can see that the CART Model correctly predict the letter 88% of the
times.

### Building a Random Forest Model

``` r
set.seed(123)
RFM = randomForest(letter ~ .-isB, data = train1)
```

``` r
predRFM = predict(RFM, newdata = test1)
x = table(test1$letter, predRFM)
kable(x)
```

|     |   A |   B |   P |   R |
|:----|----:|----:|----:|----:|
| A   | 395 |   0 |   0 |   0 |
| B   |   0 | 378 |   0 |   5 |
| P   |   0 |   3 | 397 |   1 |
| R   |   0 |  10 |   1 | 368 |

Let’s see the accuracy of the model.

``` r
#Accuracy 
sum(diag(x))/sum(x)
```

    ## [1] 0.987163

We can see that the random forest model have an accuracy of ***98.7%***.
