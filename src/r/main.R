# Titanic survival prediction with base R logistic regression.
# Download train.csv and test.csv from the Kaggle Titanic competition.
# Both files belong in src/data/ and must not be committed to GitHub.

data_dir <- Sys.getenv("TITANIC_DATA_DIR", unset = "src/data")
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")

cat("=== Titanic Survival Prediction (R) ===\n")
cat("Looking for data in:", data_dir, "\n")

if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("Missing train.csv or test.csv. Download both from Kaggle and place them in src/data/.")
}

train <- read.csv(train_path, stringsAsFactors = FALSE, na.strings = c("", "NA"))
test <- read.csv(test_path, stringsAsFactors = FALSE, na.strings = c("", "NA"))

cat("\n[1] Loaded train.csv:", nrow(train), "rows,", ncol(train), "columns\n")
cat("    Loaded test.csv: ", nrow(test), "rows,", ncol(test), "columns\n")
cat("    Survival labels in train.csv:\n")
print(table(train$Survived))

numeric_features <- c("Pclass", "Age", "Fare", "FamilySize")
categorical_features <- c("Sex", "Embarked")
original_features <- c("Pclass", "Age", "Fare", "Sex", "Embarked")

cat("\n[2] Missing values in the original training features:\n")
print(colSums(is.na(train[, original_features, drop = FALSE])))

add_family_size <- function(df) {
  df$FamilySize <- df$SibSp + df$Parch + 1
  df
}

train <- add_family_size(train)
test <- add_family_size(test)

cat("\n[3] Created FamilySize = SibSp + Parch + 1 in both datasets.\n")
cat("    Selected features:", paste(c(numeric_features, categorical_features),
                                    collapse = ", "), "\n")
cat("    Numeric missing values -> training-set median; categorical -> most frequent.\n")
cat("    Categorical variables are dummy-coded automatically by glm().\n")

# Imputation statistics and factor levels come ONLY from the training partition.
preprocess <- function(df, reference) {
  for (column in numeric_features) {
    reference_median <- median(reference[[column]], na.rm = TRUE)
    df[[column]][is.na(df[[column]])] <- reference_median
  }

  for (column in categorical_features) {
    reference_values <- as.character(reference[[column]])
    reference_values <- reference_values[!is.na(reference_values) &
                                         reference_values != ""]
    most_frequent <- names(sort(table(reference_values), decreasing = TRUE))[1]
    allowed_levels <- sort(unique(reference_values))

    values <- as.character(df[[column]])
    values[is.na(values) | values == "" | !(values %in% allowed_levels)] <-
      most_frequent
    df[[column]] <- factor(values, levels = allowed_levels)
  }

  df
}

model_formula <- Survived ~ Pclass + Age + Fare + FamilySize + Sex + Embarked
to_class <- function(probability) as.integer(probability >= 0.5)
accuracy <- function(actual, predicted) mean(as.integer(actual) == predicted)

# Stratified split; 712 training rows and 179 held-out rows for Kaggle's dataset.
set.seed(42)
training_indices <- sort(unlist(lapply(
  split(seq_len(nrow(train)), train$Survived),
  function(indices) sample(indices, floor(0.8 * length(indices)))
)))
holdout_indices <- setdiff(seq_len(nrow(train)), training_indices)

training_subset <- train[training_indices, , drop = FALSE]
holdout_subset <- train[holdout_indices, , drop = FALSE]

training_ready <- preprocess(training_subset, training_subset)
holdout_ready <- preprocess(holdout_subset, training_subset)

model <- glm(model_formula, data = training_ready, family = binomial())
training_predictions <- to_class(predict(model, newdata = training_ready,
                                         type = "response"))
holdout_predictions <- to_class(predict(model, newdata = holdout_ready,
                                        type = "response"))

cat("\n[4] Logistic regression evaluated with an 80/20 stratified split:\n")
cat("    Training rows:", nrow(training_ready),
    "| Held-out rows:", nrow(holdout_ready), "\n")
cat(sprintf("    Training accuracy: %.4f (%.2f%%)\n",
            accuracy(training_ready$Survived, training_predictions),
            100 * accuracy(training_ready$Survived, training_predictions)))
cat(sprintf("    Held-out accuracy: %.4f (%.2f%%)\n",
            accuracy(holdout_ready$Survived, holdout_predictions),
            100 * accuracy(holdout_ready$Survived, holdout_predictions)))

# Refit on all labeled training rows for predictions on Kaggle's unlabeled test.csv.
full_training_ready <- preprocess(train, train)
test_ready <- preprocess(test, train)
final_model <- glm(model_formula, data = full_training_ready, family = binomial())
full_predictions <- to_class(predict(final_model,
                                    newdata = full_training_ready,
                                    type = "response"))
test_predictions <- to_class(predict(final_model, newdata = test_ready,
                                    type = "response"))
preview <- data.frame(PassengerId = test$PassengerId,
                      Survived = test_predictions)

cat("\n[5] Refit model on all labeled train.csv rows.\n")
cat(sprintf("    Full training-data accuracy (in-sample): %.4f\n",
            accuracy(train$Survived, full_predictions)))
cat("    Generated", length(test_predictions),
    "predictions for Kaggle test.csv.\n")
cat("    Predicted survivors:", sum(test_predictions), "\n")
cat("    First 10 predictions (PassengerId, Survived):\n")
print(head(preview, 10), row.names = FALSE)

cat("\n[6] Kaggle test.csv does not contain true Survived labels.\n")
cat("    Its accuracy cannot be computed from these two files.\n")
cat("    The held-out accuracy above is the out-of-sample evaluation estimate.\n")
cat("\nR pipeline completed successfully.\n")
