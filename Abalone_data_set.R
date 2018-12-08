# Predicting the age of abalone from physical measurements. 
# The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age. Further information, such as weather patterns and location (hence food availability) may be required to solve the problem.

# Loading data
library(caret)
library(corrplot)
library(dplyr)
library(rpart)
library(pROC)
original_data <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data', sep=',')
AbaloneDF = original_data

# We rename the columns by using the following information given from the website:

# Sex / nominal / -- / M, F, and I (infant) 
# Length / continuous / mm / Longest shell measurement 
# Diameter	/ continuous / mm / perpendicular to length 
# Height / continuous / mm / with meat in shell 
# Whole weight / continuous / grams / whole abalone 
# Shucked weight / continuous	/ grams / weight of meat 
# Viscera weight / continuous / grams / gut weight (after bleeding) 
# Shell weight / continuous / grams / after being dried 
# Rings / integer / -- / +1.5 gives the age in years 

names(AbaloneDF) = c('Sex', 'Length', 'Diameter', 'Height', 
                     'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight',
                     'Rings')

#### Response is 'Rings' and it is integer. So we will use either classification or regression techniques to solve the problem.

###################### Regression Technique
str(AbaloneDF)
summary(AbaloneDF)
plot(AbaloneDF$Rings)

# No missing values.

outcomeName <- 'Rings'

AbaloneDF$Sex = as.numeric(AbaloneDF$Sex) - 1

# Visualize the data and check the necessity of scaling or transforming data

pairs20x(AbaloneDF)

# 'Height' is possibly needed to be scaled
# We notice that 'weight' have 4 variables linearly associated and we want to know how much they are correlated to each other. 


head(sort(AbaloneDF$Height, decreasing = T)) # 2 outliers in Height. We can simply remove from the dataset.

AbaloneDF = AbaloneDF[-which(AbaloneDF$Height == max(AbaloneDF$Height)), ] # remove the row where the 'Height' is max
AbaloneDF = AbaloneDF[-which(AbaloneDF$Height == max(AbaloneDF$Height)), ] # remove the row where the 'Height' is the 2nd max
hist(AbaloneDF$Height)

cors = cor(AbaloneDF[, 1:9])[1:8, 9] # correlation to each other

correlations = cor(AbaloneDF[, 1:9])
corrplot(correlations) # 'Sex' does not contribute to the prediction
AbaloneDF$Sex = NULL

anova.fit = lm(Rings~., data = AbaloneDF)
summary(anova.fit) # Notice that no evidence against that 'Length' is not significant.
AbaloneDF$Length = NULL

lm(AbaloneDF$Whole_weight ~ AbaloneDF$Shucked_weight + AbaloneDF$Viscera_weight + AbaloneDF$Shell_weight) %>% summary

# Remind that our output is integer
# Since we have colinearity between variables. Can include Whole_weight only for prediction
AbaloneDF <- AbaloneDF[, c('Diameter', 'Height', 'Whole_weight', 'Rings')]

# Models can be used
names(getModelInfo())

###### Method 1: Use Penalty regression (glmnet)
set.seed(20181208)
splitIndex <- createDataPartition(AbaloneDF[, "Rings"], p = .75, list = FALSE, times = 1)
trainDF <- AbaloneDF[splitIndex, ]
testDF  <- AbaloneDF[-splitIndex, ]
predictorsNames <- names(AbaloneDF)[names(AbaloneDF) != outcomeName]

objControl_glmnet <- trainControl(method='cv', number=5, returnResamp='none', verboseIter = F)
objModel_glmnet <- train(trainDF[,predictorsNames], 
                         trainDF[, outcomeName], 
                         method='glmnet',  
                         trControl=objControl_glmnet
                         )

# See what we obtain from the trained model

summary(objModel_glmnet)

# Make predictions and Use MSE as loss function

predictions_glmnet <- predict(object=objModel_glmnet, testDF[, predictorsNames])
mean(testDF[,outcomeName] - round(predictions_glmnet))^2 # MSE = 0.001128


###### Method 2: Use Neural Network (nnet) with one hidden layer
objModel_nnet <- train(Rings ~ ., 
                       data = trainDF,
                       method = "nnet", 
                       maxit = 100, 
                       linout = 1,
                       trace = F,
                       tuneGrid = expand.grid(.decay = c(0.5, 0.1), .size = 3)
                       )
                       
# See what we obtain from the trained model

summary(objModel_nnet)

# Make predictions and Use MSE as loss function

predictions_nnet <- predict(object=objModel_nnet, testDF[, predictorsNames])
mean(testDF[, outcomeName] - round(predictions_nnet))^2 # MSE = 7.460185e-05



###### Method 3: Use Neural Network (nnet) with 3 layers
objModel_nnet3 <- train(Rings ~ ., 
                       data = trainDF,
                       method = "nnet", 
                       maxit = 100, 
                       linout = 1,
                       trace = F,
                       tuneGrid = expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7)))

# Make predictions and Use MSE as loss function

predictions_nnet3 <- predict(object=objModel_nnet3, testDF[, predictorsNames])
mean(testDF[, outcomeName] - round(predictions_nnet3))^2 # MSE = 8.289094e-06


