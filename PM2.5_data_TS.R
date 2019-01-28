


library(e1071)
library(caret)
library(dplyr)

# Loading data
rawdata <- read.csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv')

# No: row number 
# year: year of data in this row 
# month: month of data in this row 
# day: day of data in this row 
# hour: hour of data in this row 
# pm2.5: PM2.5 concentration (ug/m^3) 
# DEWP: Dew Point (â„ƒ) 
# TEMP: Temperature (â„ƒ) 
# PRES: Pressure (hPa) 
# cbwd: Combined wind direction 
# Iws: Cumulated wind speed (m/s) 
# Is: Cumulated hours of snow 
# Ir: Cumulated hours of rain 

rawdata$No = NULL
summary(rawdata)



# ----- Imputation function begin # https://github.com/mxie622/R-tools-/blob/master/IMPUTATION_numeric_KNN.R
KNN_imputation_numeric = function(df, outcome, variables, k){
  # df : Raw dataset; data.frame
  # outcome : 'variable' of 'df' to be filled; numeric variable 
  # variables: predictors # Be careful when filling multiple NA from different columns
  # k : The number of nearest neighbour; a number or a sequence of possible k
  
  # outcome <- 'Sepal.Length' # To fill for example
  
  df$index_mikexie = 1 : nrow(df)
  if (missing(variables)){
    variables = names(df[!names(df) %in% c(outcome)])
    variables = variables[-length(variables)]
  }
  
  f <- as.formula(
    paste(outcome, 
          paste(variables, collapse = " + "), 
          sep = " ~ "))
  
  library(caret)
  # Split raw data into 2 sets
  No_NA_df <- df[complete.cases(df), ] # Complete set with no NA
  NA_df <- df[!complete.cases(df), ] # Take a subset
  selecting_k <- train(form = f,
                       method = "knn",
                       data = No_NA_df,
                       tuneGrid = expand.grid(k = k)) # Select the best k
  
  best_k = selecting_k$results$k[which(selecting_k$results$RMSE == min(selecting_k$results$RMSE))] # Select the best k with minimum RMSE
  model_to_fill_NA <- knnreg(formula = f, data = No_NA_df, k = best_k)
  
  temp0 <- NA_df[rowSums(is.na(NA_df[!names(NA_df) %in% c(outcome)])) == 0, ][variables]
  
  NA_df[rowSums(is.na(NA_df[!names(NA_df) %in% c(outcome)])) == 0, ][, outcome] = predict(model_to_fill_NA, temp0)
  
  new_df = rbind(NA_df, No_NA_df)
  new_df = new_df[order(new_df$index_mikexie), ]
  new_df$index_mikexie = NULL
  new_df
}

# ----- Imputation function end

rawdata <- KNN_imputation_numeric(rawdata, 'pm2.5', k = 3)
summary(rawdata) # Filling done

splitIndex <- 1:(nrow(rawdata) / 2)
df_train <- rawdata[splitIndex, ] # df : imputation set; df_train : Raw training set
df_test  <- rawdata[-splitIndex, ] # Testing set for imputation quality

# ---- TS modelling

df_train.ts = ts(df_train$pm2.5, start = 2010,frequency=24)
plot(df_train.ts, main = "CO2(ppm) recorded from 2014 to 2016", xlab = "HOUR", ylab="PM2.5")

# model_1 ARIMA(0, 1, 1) x (0, 1, 1)
fit1 = arima(df_train.ts, order = c(0, 1, 1), seasonal = list(order = c(0, 1, 1), period = 24))
fit1 # aic = 209268.4
pacf(residuals(fit1))
fit1.pred <- predict(fit1, n.ahead = nrow(df_test))
fit1.RMSEP = sqrt(1/4*sum((df_test$pm2.5 - fit1.pred$pred)^2))
fit1.RMSEP # 16445.79

# model_2 ARIMA(1, 1, 1) x (1, 1, 1)
fit2 = arima(df_train.ts, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 1), period = 24))
fit2 # aic = 209257.2
pacf(residuals(fit2))
fit2.pred <- predict(fit2, n.ahead = nrow(df_test))
fit2.RMSEP = sqrt(1/4*sum((df_test$pm2.5 - fit2.pred$pred)^2))
fit2.RMSEP #  16244.78

# model_3 ARIMA(1, 0, 1) x (1, 0, 1)
fit3 = arima(df_train.ts, order = c(1, 0, 1), seasonal = list(order = c(1, 0, 1), period = 24))
fit3 # aic = 208773.2
pacf(residuals(fit3))
fit3.pred <- predict(fit3, n.ahead = nrow(df_test))
fit3.RMSEP = sqrt(1/4*sum((df_test$pm2.5 - fit3.pred$pred)^2))
fit3.RMSEP #  6853.29

# model_4 ARIMA(1, 0, 0) x (1, 0, 0)
fit4 = arima(df_train.ts, order = c(1, 0, 0), seasonal = list(order = c(1, 0, 0), period = 24))
fit4 # aic = 209004.4
pacf(residuals(fit4))
fit4.pred <- predict(fit4, n.ahead = nrow(df_test))
fit4.RMSEP = sqrt(1/4*sum((df_test$pm2.5 - fit4.pred$pred)^2))
fit4.RMSEP #  6856.662


# ------- XGboost modelling
library(xgboost);
library(onehot)
library(quantmod); library(TTR);

str(df_train)
# One-hot 
dmy <- dummyVars(" ~ .", data = rawdata)
trsf <- data.frame(predict(dmy, newdata = rawdata))
trsf <- as.matrix(trsf)

splitIndex <- 1:(nrow(rawdata) / 2)
df_train <- trsf[splitIndex, ] # df : imputation set; df_train : Raw training set
df_test  <- trsf[-splitIndex, ] # Testing set for imputation quality
colnames(df_test)

X_train = df_train[, c(1:4, 6:ncol(df_train))] # The 5th variable is the response 'pm2.5'
Y_train = df_train[, "pm2.5"]
X_test = df_test[, c(1:4, 6:ncol(df_test))]
Y_test = df_test[, "pm2.5"]

dtrain = xgb.DMatrix(data = X_train, label = Y_train)
fit5_xgModel = xgboost(data = dtrain, nround = 25)

# Try CV
cv = xgb.cv(data = dtrain, nround = 25, nfold = 5)
# 

XG_preds = predict(fit5_xgModel, X_test)
fit5.XG.RMSEP = sqrt(1/4*sum((Y_test - XG_preds)^2))
fit5.XG.RMSEP # 5086.488

cv.preds = predict(cv, X_test)
xgb.importance(model = xgModel) # Print the importance of variables


# --------- End
