# loading relevant packages
library(tidyverse)
library(readxl)
library(xgboost)
library(caret)
library(foreach)
library(doParallel)

# reading data into dataframe
pitches <- read_xlsx("data.xlsx")

# selecting and renaming relevant columns while eliminating pitcher side information
pitch_data <- pitches |>
  select(PID, PITCHER_KEY, THROW_SIDE_KEY, PITCH_TYPE_TRACKED_KEY, INDUCED_VERTICAL_BREAK,
         HORIZONTAL_BREAK, SPIN_RATE_ABSOLUTE, RELEASE_SPEED, HORIZONTAL_APPROACH_ANGLE, 
         VERTICAL_APPROACH_ANGLE, PLATE_X, PLATE_Z) |>
  rename(VERT_BREAK = INDUCED_VERTICAL_BREAK,
         HOR_BREAK = HORIZONTAL_BREAK,
         SPIN_RATE = SPIN_RATE_ABSOLUTE, 
         HOR_ANGLE = HORIZONTAL_APPROACH_ANGLE,
         VERT_ANGLE = VERTICAL_APPROACH_ANGLE,
         ) |> 
  mutate(HOR_BREAK = abs(HOR_BREAK), 
         HOR_ANGLE = abs(HOR_ANGLE))
  

# calculating outliers in the top/bottom 10% of eight variables to identify high
# dew point likelihood

# defining the threshold or Z-score for identifying outliers
outlier_threshold <- 1.28

# specifying columns for outlier detection
cols_to_process <- c("VERT_BREAK", "HOR_BREAK", "SPIN_RATE", "RELEASE_SPEED", 
                     "HOR_ANGLE", "VERT_ANGLE", "PLATE_X", "PLATE_Z")

# iterating through each column to calculate Z-scores and flag outliers
for (col in cols_to_process) {
  z_col <- paste0(col, "_Z")
  outlier_col <- paste0(col, "_outlier")
  
  pitch_data <- pitch_data  |> 
    arrange(PITCHER_KEY, PITCH_TYPE_TRACKED_KEY) |>
    group_by(PITCHER_KEY, PITCH_TYPE_TRACKED_KEY) |>
    mutate(
      !!z_col := abs(scale(!!sym(col))) |> coalesce(0),
      !!outlier_col := ifelse(!!sym(z_col) > outlier_threshold, 1, 0)
    ) 
}

# calculating percentage of outliers across specified columns for each pitch
pitch_data <- pitch_data |>
  rowwise() |>
  mutate(
    outlier_percentage = sum(c_across(ends_with("_outlier"))) / length(cols_to_process)
  ) |>
  ungroup()

# selecting relevant features and target variable for XGBoost model
pitch_data <- pitch_data |> 
  select(PID, VERT_BREAK, HOR_BREAK, SPIN_RATE, RELEASE_SPEED, HOR_ANGLE, VERT_ANGLE,
         PLATE_X, PLATE_Z, outlier_percentage)

# setting seed and splitting data
set.seed(2023)
train_index <- createDataPartition(pitch_data$outlier_percentage, p = 0.7, list = FALSE)
train_data <- pitch_data[train_index, ]
test_data <- pitch_data[-train_index, ]

# training an XGBoost model
xgb_model <- xgboost(data = as.matrix(train_data[, !(names(train_data) %in% c("outlier_percentage", "PID"))]), 
                     label = train_data$outlier_percentage,
                     objective = "reg:squarederror",
                     nrounds = 100,
                     max_depth = 3,
                     eta = 0.1,
                     nthread = 7)  

# generating predictions from trained XGBoost model
predictions <- predict(xgb_model, newdata = as.matrix(test_data[, !(names(test_data) %in% c("outlier_percentage", "PID"))]))

# calculating MSE, RMSE, and R-squared for model evaluation  
mse <- mean((test_data$outlier_percentage - predictions)^2)
rmse <- sqrt(mse)
rsquared <- 1 - (sum((test_data$outlier_percentage - predictions)^2) / sum((test_data$outlier_percentage - mean(test_data$outlier_percentage))^2))

# combining test and train data and generating predictions from the model
combined_data <- rbind(train_data, test_data)

predictions_combined <- predict(xgb_model, as.matrix(combined_data[, !(names(combined_data) %in% c("outlier_percentage", "PID"))]))

combined_data$Predicted_Outlier_Percentage <- predictions_combined

actual_values <- combined_data$outlier_percentage
predicted_values <- combined_data$Predicted_Outlier_Percentage

# calculating R-squared from combined data
r_squared <- 1 - sum((actual_values - predicted_values)^2) / sum((actual_values - mean(actual_values))^2)

# preparing for K-Fold cross validation
num_cores <- 7
cl <- makeCluster(num_cores)
registerDoParallel(cl)
num_folds <- 5

# defining control parameters for cross-validation and defining the model function 
ctrl <- trainControl(method = "cv", number = num_folds)

# Defining a function for building the XGBoost model
xgb_model_build <- function(train_data, ctrl) {
  library(xgboost)
  library(caret)  
  
  xgb_model <- xgboost(data = as.matrix(train_data[, !(names(train_data) %in% c("outlier_percentage", "PID"))]),
                       label = train_data$outlier_percentage,
                       objective = "reg:squarederror",
                       nrounds = 100,
                       max_depth = 3,
                       eta = 0.1,
                       nthread = 7)
  return(xgb_model)
}

# loading the 'caret' library in the parallel cluster
clusterEvalQ(cl, library(caret))

# initializing the data frame for cross-validation results
results <- data.frame(Fold = integer(), Performance = double())

# performing cross-validation for each fold
foreach(fold = 1:num_folds) %do% {
  set.seed(fold)
  train_index <- createDataPartition(pitch_data$outlier_percentage, p = 0.7, list = FALSE)
  train_data <- pitch_data[train_index, ]
  test_data <- pitch_data[-train_index, ]
  
  fold_model <- xgb_model_build(train_data, ctrl)
  
  predictions <- predict(fold_model, newdata = as.matrix(test_data[, !(names(test_data) %in% c("outlier_percentage", "PID"))]))
  
  # calculating the MSE for the fold
  performance_metrics <- mean((test_data$outlier_percentage - predictions)^2)
  
  # Appending the result to the 'results' data frame
  results <- rbind(results, data.frame(Fold = fold, Performance = performance_metrics))
}

# storing the final results while stopping the parallel cluster
final_results <- results

stopCluster(cl)

# creating hyperparameter grid and initializing best params/rmse
param_grid <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.1, 0.3),
)

best_params <- NULL
best_rmse <- Inf

train_data_x <- train_data[, !(names(train_data) %in% c("outlier_percentage", "PID"))]
test_data_x <- test_data[, !(names(test_data) %in% c("outlier_percentage", "PID"))]

# iterating through each hyperparameter to find best RMSE
for (i in 1:nrow(param_grid)) {
  param_set <- param_grid[i, ]
  xgb_model <- xgboost(data = as.matrix(train_data_x), label = train_data$outlier_percentage,
                       objective = "reg:squarederror",
                       nrounds = param_set$nrounds,
                       max_depth = param_set$max_depth,
                       eta = param_set$eta,
                       nthread = 7)
  
  predictions <- predict(xgb_model, as.matrix(test_data_x))
  rmse <- sqrt(mean((test_data$outlier_percentage - predictions)^2))
  
  if (rmse > best_rmse) {
    best_rmse <- rmse
    best_params <- param_set
  }
}

# training XGBoost model based on  best hyperparameters 
final_xgb_model <- xgboost(data = as.matrix(train_data_x), 
                           label = train_data$outlier_percentage,
                           objective = "reg:squarederror",
                           nrounds = best_params$nrounds,
                           max_depth = best_params$max_depth,
                           eta = best_params$eta,
                           nthread = 7)

test_predictions <- predict(final_xgb_model, as.matrix(test_data_x))
test_rmse <- sqrt(mean((test_data$outlier_percentage - test_predictions)^2))
test_rsquared <- 1 - (sum((test_data$outlier_percentage - predictions)^2) / sum((test_data$outlier_percentage - mean(test_data$outlier_percentage))^2))

# combining test and train data and generating predictions from the model
combined_data <- rbind(train_data, test_data)
predictions_combined <- predict(final_xgb_model, as.matrix(combined_data[, !(names(combined_data) %in% c("outlier_percentage", "PID"))]))
combined_data$Predicted_Outlier_Percentage <- predictions_combined
actual_values <- combined_data$outlier_percentage
predicted_values <- combined_data$Predicted_Outlier_Percentage

# Calculating R-squared of the combined data 
final_r_squared <- 1 - sum((actual_values - predicted_values)^2) / sum((actual_values - mean(actual_values))^2)

# formatting and exporting dewpoint probability to submission CSV file
combined_data <- combined_data |> 
  select(PID, Predicted_Outlier_Percentage) |> 
  rename(DEWPOINT_AFFECTED = Predicted_Outlier_Percentage)

write.csv(combined_data, file = "submission.csv", row.names = FALSE)
