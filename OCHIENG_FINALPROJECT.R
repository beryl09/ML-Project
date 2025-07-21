#Final project
da = read.csv("depression.csv", header = T) 
# Remove the year variable
da <- da[, !names(da) %in% c("year")]
dim(da) #6,000 by 12
str(da)
da$mdeSI = factor(da$mdeSI)
da$income = factor(da$income)
da$gender = factor(da$gender, levels = c("Male", "Female")) #Male is the reference group
da$age = factor(da$age)
da$race = factor(da$race, levels = c("White", "Hispanic", "Black", "Asian/NHPIs", "Other")) #white is the reference
da$insurance = factor(da$insurance, levels = c("Yes", "No")) #"Yes" is the reference group
da$siblingU18 = factor(da$siblingU18, levels = c("Yes", "No"))
da$fatherInHH = factor(da$fatherInHH)
da$motherInHH = factor(da$motherInHH)
da$parentInv = factor(da$parentInv)
da$schoolExp = factor(da$schoolExp, levels = c("good school experiences", "bad school experiences"))
(n = dim(da)[1])
set.seed(2024)
index = sample(1:n, 4500) #75% of training and 25% of test data
train = da[index,] 
test = da[-index,]
dim(train)
dim(test)

#Data Exploration and Features Selection
summary(train)  # Summary statistics for numerical variables
table(train$mdeSI)  # Distribution of target variable
# Feature Selection
# Recursive Feature Elimination (Example with Logistic Regression)
library(caret)
ctrl <- rfeControl(functions=rfFuncs, method="cv", number=10)



#(A) Classifier one; Logistic model
model_logit <-glm(mdeSI ~ ., family=binomial(link="logit"), da=train) # Train logistic regression model using selected features
summary(model_logit)# Summary of the logistic regression model

odds_ratios <- exp(coef(model_logit))# Calculate odds ratios for each predictor
conf_intervals <- confint(model_logit) #Confidence intervals
odds_ratios <- exp(coef(model_logit))# Compute odds ratios
lower_bounds <- exp(conf_intervals[, 1])
upper_bounds <- exp(conf_intervals[, 2])# Compute AOR lower and upper bounds
variables <- names(coef(model_logit))# Extract variable names

# Combine AORs with variable names and their confidence intervals
(result <- data.frame(Predictor_Variable = variables, 
                      AOR = odds_ratios,
                      Lower_Bound = lower_bounds,
                      Upper_Bound = upper_bounds))

prob.train = predict(model_logit, train, type = "response") #: for Logit that is model in my case
pred.train = rep(0, 4500)
pred.train[prob.train > 0.5] = 1
(CM.train = table(train$mdeSI, pred.train)) #confusion matrix
(accuracy.train = (CM.train[1,1] + CM.train[2,2])/4500)
(recall.train = CM.train[2,2]/sum(CM.train[2,]))  #denominator: all the true "1"
(precision.train = CM.train[2,2]/sum(CM.train[,2])) #denominator: all the predicted "1"

prob.test = predict(model_logit, test, type = "response") #: for Probit
pred.test = rep(0, 1500)
pred.test[prob.test > 0.5] = 1
(CM.test = table(test$mdeSI, pred.test)) #confusion matrix
(accuracy.test = (CM.test[1,1] + CM.test[2,2])/1500)
(recall.test = CM.test[2,2]/sum(CM.test[2,]))  #denominator: all the true "1"
(precision.test = CM.test[2,2]/sum(CM.test[,2])) #denominator: all the predicted "1"

# install required packages for ROC AND AUC 
install.packages("ROCR")
library(ROCR)

ROCPred<- prediction(prob.test, test$mdeSI)# Create a prediction object for ROCR
ROCPer <- performance(ROCPred, measure = "tpr", x.measure = "fpr")# Create an ROC curve object
plot(ROCPer, main = "ROC Curve", col = "blue", lwd = 2)# Plot the ROC curve
legend("bottomright", legend = 
         paste("AUC =", round(performance(ROCPred, measure = "auc")
                              @y.values[[1]], 2)), col = "blue", lwd = 2)
# Find optimal threshold using Youden's J statistic
optimal.threshold <- coords(ROCPred, "best", ret="threshold", transpose=FALSE)
optimal.threshold
# Add labels and a legend to the plot
# Load required libraries
library(ggplot2)
library(dplyr)
# Calculate accuracy and recall for different thresholds in the specified range
thresholds <- seq(0.05, 0.5, by = 0.01)
metrics <- sapply(thresholds, function(threshold) {
  predicted_labels <- ifelse(prob.test > threshold, 1, 0)
  CM <- table(test$mdeSI, predicted_labels)
  if (nrow(CM) < 2 || ncol(CM) < 2) {
    return(c(Threshold = threshold, Accuracy = 0, Recall = 0))
  }
  accuracy <- sum(diag(CM)) / sum(CM)
  recall <- CM[2, 2] / sum(CM[2, ])
  return(c(Threshold = threshold, Accuracy = accuracy, Recall = recall))
})

# Convert metrics to a dataframe
metrics_df <- as.data.frame(t(metrics))

# Calculate the difference between accuracy and recall rates
metrics_df <- metrics_df %>%
  mutate(Difference = Accuracy - Recall)
# Plot the difference between accuracy and recall rates against different thresholds
ggplot(metrics_df, aes(x = Threshold)) +
  geom_line(aes(y = Accuracy, color = "Accuracy"), size = 1.5) +
  geom_line(aes(y = Recall, color = "Recall"), size = 1.5) +
  scale_color_manual(values = c("Accuracy" = "#1f78b4", "Recall" = "#e31a1c")) +
  geom_vline(xintercept = thresholds[which.max(metrics_df$Accuracy)], linetype = "dashed", color = "#1f78b4", size = 1) +
  geom_vline(xintercept = thresholds[which.max(metrics_df$Recall)], linetype = "dashed", color = "#e31a1c", size = 1) +
  labs(title = "Accuracy and Recall vs. Threshold",
       x = "Threshold", y = "Rate",
       caption = "Dashed lines indicate optimal thresholds") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 20, face = "bold"),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 14),
    panel.border = element_rect(color = "black", fill = NA, size = 1) # Add border around the plot
  ) +
  coord_cartesian(ylim = c(0, 1)) # Set y-axis range





#(B) Classifier two
#install.packages("randomForest")
library(ggplot2)
library(randomForest)

# Train a Random Forest model on the training data
model <- randomForest(mdeSI ~ ., data = train, mtry = 3, ntree = 500, importance =T)
model
# Checking the important features
importance(model)
varImpPlot(model)
plot(model)
model$confusion
model$specificity
#Validate the model using test data
library(caret)
model1 = predict(model, test)
confusionMatrix(model1, test$mdeSI)

head(model$err.rate)
oob.error.data <- data.frame(
  Trees=rep(1:nrow(model$err.rate), times=3),
  Type=rep(c("OOB", "Not mdeSI", "mdeSI"), each=nrow(model$err.rate)),
  Error=c(model$err.rate[,"OOB"],
          model$err.rate[,"No"],
          model$err.rate[,"Yes"]))

#To check the optimal subset of features for each split
oob.values <- vector(length=10)
for(i in 1:10) {
  model <- randomForest(mdeSI ~ ., data=da, mtry=i, ntree=500)
  oob.values[i] <- model$err.rate[nrow(model$err.rate),1]
}
oob.values


















