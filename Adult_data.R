library(caret)
library(corrplot)
library(dplyr)
library(rpart)
library(stringr)
library(s20x)
library(ggplot2)
library(pROC)
library(mctest)
#---------- This is a binary classification problem. Task: predict a person's {income => 50K or not} a year
# Loading data
original_data_train <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', sep=',')
original_data_test = read.delim('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')

trainDF = original_data_train
testDF = as.data.frame(original_data_test)

str(trainDF)
str(testDF)

# Predictor information given below:

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# Rename these two datasets

names(trainDF) = c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                   'marital_status', 'occupation', 'relationship', 'race', 'sex',
                   'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')


outcomeName = 'income'
predictorsNames =  names(trainDF)[names(trainDF) != outcomeName]


testDF = str_split_fixed(testDF$X.1x3.Cross.validator, ',', ncol(trainDF))
testDF = as.data.frame(testDF) 
names(testDF) = names(trainDF) # replace the name of the old dataset by the ''trainDF''

# Check which columns have missing values

summary(trainDF) # 'workclass(factor)', 'occupation(factor)', 'native-country(factor)'
summary(testDF) # 'workclass(factor)', 'occupation(factor)', 'native-country(factor)'

# There are some outliers in 'capital-gain' and 'capital-loss' which are not reasonable.

# Due to many categorical variables, we think of use randomForest to make classification.

#### Feather Engineering 
# Make all variables of ''testDF'' consistent to ''trainDF''
for (i in 1 : ncol(testDF)){
  if (class(trainDF[,i]) == 'integer'){
    testDF[, i] = as.numeric(testDF[, i])
  }
}

# Though '?' indicates a missing value, we can still regard it as a type and make prediction.

levels(trainDF$`native-country`)
levels(testDF$`native-country`)


# Notice that native-country in ''trainDF'' has 'Holand-Netherlands' but ''testDF'' does not have

# Visualize the data. (The number of observations is too many to plot all variables together in a screen. So here plot one by one)
# Use histogram to see the distribution of continous variables and normal plot for categorical variables
hist(trainDF[, 1])
plot(trainDF[, 2])
hist(trainDF[, 3]) # Potential outliers
plot(trainDF[, 4]) 
hist(trainDF[, 5])
plot(trainDF[, 6])
plot(trainDF[, 7])
plot(trainDF[, 8])
plot(trainDF[, 9])
plot(trainDF[, 10])
hist(trainDF[, 11]) # Potential outliers
hist(trainDF[, 12]) # Potential outliers
hist(trainDF[, 13]) # Potential outliers
plot(trainDF[, 14]) 
plot(trainDF[, 15])

# Now remove outliers

sort(trainDF[, 3], decreasing = T)
sort(trainDF[, 11], decreasing = T)
sort(trainDF[, 12], decreasing = T)
sort(trainDF[, 13], decreasing = T)

# The 11th variable, 'capital-gain' has many extremely high values.
# Transform the data by log for both training and testing sets

new_trainDF <- trainDF
new_trainDF[, 3] = log(new_trainDF[, 3] + 0.01) # avoid log(0)
new_trainDF[, 11] = log(new_trainDF[, 11] + 0.01) # avoid log(0)
new_trainDF[, 12] = log(new_trainDF[, 12] + 0.01) # avoid log(0)
new_trainDF[, 13] = log(new_trainDF[, 13] + 0.01) # avoid log(0)

new_testDF <- testDF
new_testDF[, 3] = log(new_testDF[, 3] + 0.01) # avoid log(0)
new_testDF[, 11] = log(new_testDF[, 11] + 0.01) # avoid log(0)
new_testDF[, 12] = log(new_testDF[, 12] + 0.01) # avoid log(0)
new_testDF[, 13] = log(new_testDF[, 13] + 0.01) # avoid log(0)


#### Dimension reduction

new_trainDF$income = as.integer(new_trainDF$income) - 1
new_testDF$income = as.integer(new_testDF$income) - 1

anova.fit = glm(formula = income ~ ., data = new_trainDF, family = binomial)
summary(anova.fit)

# Note that there are two coefficients showing NA
length(anova.fit$coefficients) > anova.fit$rank # Check collinearity between variables

# Yes. We see there exists collinearity

summary(lm(formula = education_num ~ ., 
           data = new_trainDF)) # To know the collinearity between variables

summary(glm(formula = occupation ~ ., data = new_trainDF, family = binomial))

# We found education and education_num has collinearity. 'occupation' can be dropped as well, for it is collinear

new_trainDF$education_num = NULL
new_testDF$education_num = NULL

new_trainDF$occupation = NULL
new_testDF$occupation = NULL

######### Use logistic regression for prediction: method 1

for (i in 1 : (ncol(new_trainDF) - 1)){
  new_trainDF[, i] = as.numeric(new_trainDF[, i])
  new_testDF[, i] = as.numeric(new_testDF[, i])
}

names(getModelInfo())

new_anova.fit = glm(formula = income ~ ., data = new_trainDF, 
                    family=binomial(link="logit"))


summary(new_anova.fit)
summary(anova.fit)
predictions <- predict(new_anova.fit, new_testDF[, 1 : (ncol(new_testDF) - 1)], type = 'response') # make prediction
summary(predictions)
pe_logistic = (sum(abs(round(predictions) - new_testDF$income))) / nrow(new_testDF) # prediction error
pe_logistic # 0.224


######### Use KNN   # Method 2
new_trainDF$income = as.factor(new_trainDF$income)
new_testDF$income = as.factor(new_testDF$income)


normalize <- function(x){
  y = ((x - min(x))/(max(x)-min(x)))
}

new_trainDF[, 1 : (ncol(new_trainDF) - 1)] <- as.data.frame(lapply(new_trainDF[, 1 : (ncol(new_trainDF) - 1)], normalize))
new_testDF[, 1 : (ncol(new_testDF) - 1)] <- as.data.frame(lapply(new_testDF[, 1 : (ncol(new_testDF) - 1)], normalize))

predictorsNames =  names(new_trainDF)[names(new_trainDF) != outcomeName]

tuning_parameters <- seq(3, 17, by = 2)  # Select potential values of k

knn_output = list()

# ***********Add one more factor for where 'trainDF' has a different number of factor levels from 'testDF'
addNoAnswer <- function(x){
  if(is.factor(x)) return(factor(x, levels=c(levels(x), " Holand-Netherlands")))
  return(x)
}
summary(trainDF$native_country)
new_testDF$native_country = addNoAnswer(new_testDF$native_country)
# df <- as.data.frame(lapply(df, addNoAnswer)) ####### apply the above process for all columns

# Try different k values
i = 1
for (j in tuning_parameters){
    knn_output[[i]] <- knn(new_trainDF[, 1 : (ncol(new_trainDF) -1)], 
                     new_testDF[, 1 : (ncol(new_testDF) -1)],
                     cl = new_trainDF[, ncol(new_trainDF)],
                     k = j)
    i = i + 1
}
i = 1
pe = double(length(tuning_parameters))
for (i in 1 : length(tuning_parameters)){
  tab <- table(knn_output[[i]], new_testDF[, outcomeName])
  pe[i] <- 1 - sum(diag(tab)) / sum(tab)
}
pe
min(pe) # take the minimum prediction error from K-Nearest Neighbour algorithm: 0.1886. k = 15
# {The reason I do not use <train> function is that it consumes too much time.!!!!}


######### Use naiveBayes   # Method 3
model_naiveBayes = train(new_trainDF[, predictorsNames], 
                         new_trainDF[, outcomeName], 
                         method = 'nb', 
                         trControl = trainControl(method='cv',number=10))
naiveBayes_table = prop.table(table(predict(model_naiveBayes$finalModel, 
                         new_testDF[, predictorsNames])$class, 
                 new_testDF[, outcomeName])) # table() gives frequency table, prop.table() give
pe_naiveBayes = 1 - sum(diag(naiveBayes_table)) / sum(naiveBayes_table)

pe_naiveBayes # 0.172

######### Use randomForest # Method 4

objModel_randomForest <- randomForest(income ~ ., data = new_trainDF)
predictions_rf <- predict(object = objModel_randomForest, new_testDF[, predictorsNames])
table_rf = table(predictions_rf, new_testDF[, outcomeName])
pe_rf = 1 - sum(diag(table_rf)) / sum(table_rf)
pe_rf # 0.179


