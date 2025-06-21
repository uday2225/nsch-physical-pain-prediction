tidyverse_pkgs <- c("tidyverse", "caret", "randomForest", "ROCR", "readxl", "ggplot2", "ggpubr", "reshape2")
lapply(tidyverse_pkgs, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
})

data <- read.csv("2018 NSCH_Topical_DRC_v2.csv")

selected_variables <- colnames(data)[!colnames(data) %in% c("FIPSST", "State FIPS Code")]
if (!"PHYSICALPAIN" %in% selected_variables) stop("PHYSICALPAIN outcome variable not found.")
data$PHYSICALPAIN_BIN <- ifelse(data$PHYSICALPAIN == 1, 0, 1)
data_clean <- data[, c(setdiff(selected_variables, "PHYSICALPAIN"), "PHYSICALPAIN_BIN")]
data_clean <- data_clean[complete.cases(data_clean), ]
data_clean$outcome <- as.factor(data_clean$PHYSICALPAIN_BIN)
data_clean <- data_clean[, !(names(data_clean) %in% c("PHYSICALPAIN_BIN"))]

set.seed(42)
train_index <- createDataPartition(data_clean$outcome, p = 0.8, list = FALSE)
train_data <- data_clean[train_index, ]
test_data  <- data_clean[-train_index, ]

levs <- sort(unique(as.character(train_data$outcome)))
if (length(levs) != 2) stop("Outcome must be binary for ROC/AUC.")
train_data$outcome <- factor(train_data$outcome, levels = levs, labels = c("yes", "no"))
test_data$outcome  <- factor(test_data$outcome,  levels = levs, labels = c("yes", "no"))

set.seed(42)
rf_init <- randomForest(outcome ~ ., data = train_data, importance = TRUE, ntree = 1000, sampsize = c(yes = sum(train_data$outcome == "yes"), no = sum(train_data$outcome == "no")))
importance_df <- as.data.frame(importance(rf_init))
importance_df$Variable <- rownames(importance_df)
top_vars <- importance_df %>% 
  arrange(desc(MeanDecreaseGini)) %>% 
  filter(!Variable %in% c(
    "DiffPain_18", "Diff2more_18", "cntdiff", "HEADACHE",
    "HEADACHE_CURR", "HEADACHE_DESC", "headache_18", "HeadSev_18",
    "DiffDigest_18", "HCABILITY", "DailyAct_18", "STOMACH",
    "HCEXTENT", "CondCnt27_18", "Cond2more27_18"
  )) %>% 
  slice(1:25) %>% 
  pull(Variable)
train_data_reduced <- train_data[, c(top_vars, "outcome")]
test_data_reduced <- test_data[, intersect(c(top_vars, "outcome"), colnames(test_data))]

train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10, 12))
set.seed(42)
rf_model <- train(
  outcome ~ .,
  data = train_data_reduced,
  method = "rf",
  trControl = train_control,
  metric = "ROC",
  tuneGrid = tune_grid,
  ntree = 1000,
  importance = TRUE
)

print(rf_model)

if ("SC_AGE_YEARS" %in% colnames(train_data_reduced)) {
  p1 <- ggplot(train_data_reduced, aes(x = as.numeric(SC_AGE_YEARS), fill = outcome)) +
    geom_histogram(binwidth = 1, position = "dodge") +
    labs(title = "Age Distribution by Outcome", x = "SC_AGE_YEARS", y = "count") +
    theme_minimal()
  print(p1)
}

importance_vals <- importance(rf_model$finalModel)
importance_df <- data.frame(Variable = rownames(importance_vals), importance_vals)
importance_df <- importance_df[order(importance_df$MeanDecreaseGini, decreasing = TRUE), ]
importance_df <- importance_df[1:25, ]

p_imp <- ggplot(importance_df, aes(x = MeanDecreaseGini, y = MeanDecreaseAccuracy)) +
  geom_point(color = "darkred", size = 3) +
  geom_text(aes(label = Variable), hjust = 1.1, vjust = 0.5, size = 3) +
  labs(title = "Variable Importance: Gini vs Accuracy", x = "Mean Decrease Gini", y = "Mean Decrease Accuracy") +
  theme_minimal()
print(p_imp)

rf_pred <- predict(rf_model, newdata = test_data_reduced)
rf_probs <- predict(rf_model, newdata = test_data_reduced, type = "prob")

cm <- confusionMatrix(rf_pred, test_data_reduced$outcome)
print(cm)

cm_table <- as.data.frame(cm$table)
p_cm <- ggplot(data = cm_table, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "black") +
  geom_text(aes(label = Freq), vjust = 1.5, size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal()
print(p_cm)

roc_col <- if ("yes" %in% colnames(rf_probs)) "yes" else colnames(rf_probs)[1]
pred <- prediction(rf_probs[, roc_col], test_data_reduced$outcome)
perf <- performance(pred, "tpr", "fpr")
plot(perf, col = "blue", main = "ROC Curve")
abline(a = 0, b = 1, lty = 2, col = "red")
auc <- performance(pred, "auc")@y.values[[1]]
cat(sprintf("AUC Score: %.3f\n", auc))

write.csv(train_data_reduced, "cleaned_train_data.csv", row.names = FALSE)
saveRDS(rf_model, "rf_model_top25.rds")
