################################################################
################ Financial & Risk Analytics ####################
################################################################








setwd("C:/Users/Abhishek Ranjan/Desktop/FRA Assignment")



# Reading Training Dataset and basic exploration
data <- read.csv(file = "training.csv", header = TRUE)
summary(data)
str(data)



# Formatting the data
# Converting Casenum & SeriousDlqin2yrs to Factor
data$Casenum <- as.factor(data$Casenum)
data$SeriousDlqin2yrs <- as.factor(data$SeriousDlqin2yrs)
write.csv(file = "impute_training.csv", data)









#################################################################################
########################## Data Cleaning & Preparation ##########################
#################################################################################

# Imputing NA's in NumberofDependents with average
require(plyr)
require(caret)
require(Hmisc)
imputed_value <- ceiling(mean(data$NumberOfDependents, na.rm = T))
imputed_value <- as.integer(imputed_value)
data$NumberOfDependents[is.na(data$NumberOfDependents)] <- imputed_value

# Removing the column CaseNum before Smote since it will not be included in the model
data <- data[, !(colnames(data) %in% c("Casenum"))]

attach(data)









#################################################################################
########################### Handling unbalanced data ############################
#################################################################################

# Smote the data since events(SeriousDlqin2yrs = "1") == 6%
prop.table(table(data$SeriousDlqin2yrs))
require(DMwR)
set.seed(420)
Smoted_data <- SMOTE(SeriousDlqin2yrs ~. , data, perc.over=200, perc.under=200)
prop.table(table(Smoted_data$SeriousDlqin2yrs)) # Balanced the data with events == 33%





# Rounding off Over-Sampled data for NumberOfOpenCreditLinesAndLoans && NumberOfDependents
Smoted_data$NumberOfOpenCreditLinesAndLoans <- round(Smoted_data$NumberOfOpenCreditLinesAndLoans,digits = 0)
Smoted_data$NumberOfDependents <- round(Smoted_data$NumberOfDependents,digits = 0)
Smoted_data$NumberOfOpenCreditLinesAndLoans <- as.integer(Smoted_data$NumberOfOpenCreditLinesAndLoans)
Smoted_data$NumberOfDependents <- as.integer(Smoted_data$NumberOfDependents)
write.csv(file = "Smoted_data.csv", Smoted_data)









#################################################################################
############ Building the Models for Classification on Training Data ############
#################################################################################

# Logistic regression Model
###########################
attach(Smoted_data)
logit_model <- glm(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines+NumberOfOpenCreditLinesAndLoans,
                   data=Smoted_data, family="binomial")
summary(logit_model)
anova(logit_model)

x <- predict(logit_model, Smoted_data, type = "response")
Smoted_data$Prediction <- x
Smoted_data <- within(Smoted_data, Prediction[Prediction >=0.5] <- 1)
Smoted_data <- within(Smoted_data, Prediction[Prediction <0.5] <- 0)
Smoted_data$Prediction
confusionMatrix(Smoted_data$Prediction,Smoted_data$SeriousDlqin2yrs)
summary(Smoted_data)


# Checking Area Under the curve
library(pROC)
x <- predict(logit_model, Smoted_data, type = "response")
Smoted_data$Prediction <- x
Smoted_data$SeriousDlqin2yrs <- as.integer(Smoted_data$SeriousDlqin2yrs)
auc <- roc(Smoted_data$Prediction,Smoted_data$SeriousDlqin2yrs)
print(auc)



# ROC Curve to find optimal threshold for the prediction from logit model
library(ROCR)
x <- predict(logit_model, Smoted_data, type = "response")
Smoted_data$Prediction <- x
# calculating the values for ROC curve
pred <- prediction(Smoted_data$Prediction, Smoted_data$SeriousDlqin2yrs)
perf <- performance(pred,"tpr","fpr")
# changing params for the ROC plot - width, etc
par(mar=c(5,5,2,2),xaxs = "i",yaxs = "i",cex.axis=1.3,cex.lab=1.4)
# plotting the ROC curve
plot(perf,col="black",lty=3, lwd=3)
# calculating AUC
auc <- performance(pred,"auc")
# now converting S4 class to vector
auc <- unlist(slot(auc, "y.values"))
# adding min and max ROC AUC to the center of the plot
minauc<-min(round(auc, digits = 2))
maxauc<-max(round(auc, digits = 2))
minauct <- paste(c("min(AUC)  = "),minauc,sep="")
maxauct <- paste(c("max(AUC) = "),maxauc,sep="")
legend(0.3,0.6,c(minauct,maxauct,"\n"),border="white",cex=1.7,box.col = "white")
str(perf)
plot(perf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))


# Checking cutoff alpha value
cutoffs <- data.frame(cut=perf@alpha.values[[1]], fpr=perf@x.values[[1]], tpr=perf@y.values[[1]])
head(cutoffs)
cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]
head(subset(cutoffs, fpr < 0.25))
















# Classification Tree
#####################
library(rpart)
names(Smoted_data)
attach(Smoted_data)
#Smoted_data$SeriousDlqin2yrs <- as.factor(Smoted_data$SeriousDlqin2yrs)
formula <- SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines + DebtRatio +
  NumberOfOpenCreditLinesAndLoans + NumberOfDependents
fit <- rpart(formula, method="class")
fit
printcp(fit)
plotcp(fit)
summary(fit)
plot(fit, uniform=TRUE, main="Classification Tree for Default Modeling")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# prune the tree 
pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(pfit, uniform=TRUE, 
     main="Pruned Classification Tree for Default Modeling")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)
#install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(pfit)

# Confusion Matrix of prediction on Smoted Train Dataset
x <- predict(pfit, Smoted_data, type = "class")
Smoted_data$Prediction <- x
head(Smoted_data)
names(Smoted_data)
attach(Smoted_data)
confusionMatrix(Smoted_data$Prediction,Smoted_data$SeriousDlqin2yrs)









#################################################################################
################### Evaluating model performance on Test data ###################
#################################################################################

# Reading the Test dataset
test <- read.csv(file = "test.csv", header = TRUE)




### Predicting logit model on Test Data Set ###
x <- predict(logit_model, test, type = "response")
test$Prediction <- x
test <- within(test, Prediction[Prediction >=0.35] <- 1)
test <- within(test, Prediction[Prediction <0.35] <- 0)
test$Prediction
confusionMatrix(test$Prediction,test$SeriousDlqin2yrs)





### Predicting Classification Tree on Test Data Set ###
test <- read.csv(file = "test.csv", header = TRUE)
str(test)
attach(test)
test$SeriousDlqin2yrs <- as.factor(test$SeriousDlqin2yrs)
x <- predict(pfit, newdata = test, type = "class")
test$Prediction <- x
test$Prediction
confusionMatrix(test$Prediction,test$SeriousDlqin2yrs)

