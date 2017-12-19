library(RANN)
library(LICORS)
library(moments)
library(nortest)
library(sqldf)
library(fpc)
library(neuralnet)
library(caret)
library(smbinning)

rm(list = ls())

normali_fun= function(x){
  x=(x-min(x))/(max(x)-min(x))
  return(x)
}
attr=read.csv("HR_Employee_Attrition_Data.csv")

###detail analysis of data###
describe(attr)

set.seed(223)
test = sample(1:nrow(attr),nrow(attr)/4)
nn.test = attr[test,]
nn.train = attr[-test,]

#Identifying important variables using Information value
# attr$Target[attr$Attrition == "Yes"] =1 
# attr$Target[attr$Attrition == "No"] =0
#install.packages("smbinning")
library(smbinning)
summary(nn.train)
result = smbinning(df = nn.train,y="Target",x="MonthlyIncome",p=0.05)
#result$ivtable
##MOnthlyIncome - 0.415
result1 = smbinning(df=nn.train,y="Target",x="YearsAtCompany",p=0.05)
#result1$ivtable
##YearsAtCompany - 0.3197
result2 = smbinning(df = nn.train,y="Target",x="YearsSinceLastPromotion",p=0.05)
#result2$ivtable
##YearsSinceLastPromotion - 0.0404
result3 = smbinning(df=nn.train,y="Target",x="PercentSalaryHike",p=0.05)
crosstab = table(Attrition,OverTime)
prop.table(crosstab,2)
barplot(crosstab,legend = rownames(crosstab))
chisq.test(crosstab)
#low p-value indicates OverTime is significant
crosstab1 = table(Attrition,StockOptionLevel)
chisq.test(crosstab1)
#StockOptionLevel is also significant

#Checking for collinearity among the Predictor variables using VIF
C = cor(attr$Age,attr$TotalWorkingYears)
vif_func<-function(in_frame,thresh=10,trace=T,...){
  
  require(fmsb)
  
  if(class(in_frame) != 'data.frame') in_frame<-data.frame(in_frame)
  
  #get initial vif value for all comparisons of variables
  vif_init<-NULL
  var_names <- names(in_frame)
  for(val in var_names){
    regressors <- var_names[-which(var_names == val)]
    form <- paste(regressors, collapse = '+')
    form_in <- formula(paste(val, '~', form))
    vif_init<-rbind(vif_init, c(val, VIF(lm(form_in, data = in_frame, ...))))
  }
  vif_max<-max(as.numeric(vif_init[,2]))
  
  if(vif_max < thresh){
    if(trace==T){ #print output of each iteration
      prmatrix(vif_init,collab=c('var','vif'),rowlab=rep('',nrow(vif_init)),quote=F)
      cat('\n')
      cat(paste('All variables have VIF < ', thresh,', max VIF ',round(vif_max,2), sep=''),'\n\n')
    }
    return(var_names)
  }
  else{
    
    in_dat<-in_frame
    
    #backwards selection of explanatory variables, stops when all VIF values are below 'thresh'
    while(vif_max >= thresh){
      
      vif_vals<-NULL
      var_names <- names(in_dat)
      
      for(val in var_names){
        regressors <- var_names[-which(var_names == val)]
        form <- paste(regressors, collapse = '+')
        form_in <- formula(paste(val, '~', form))
        vif_add<-VIF(lm(form_in, data = in_dat, ...))
        vif_vals<-rbind(vif_vals,c(val,vif_add))
      }
      max_row<-which(vif_vals[,2] == max(as.numeric(vif_vals[,2])))[1]
      
      vif_max<-as.numeric(vif_vals[max_row,2])
      
      if(vif_max<thresh) break
      
      if(trace==T){ #print output of each iteration
        prmatrix(vif_vals,collab=c('var','vif'),rowlab=rep('',nrow(vif_vals)),quote=F)
        cat('\n')
        cat('removed: ',vif_vals[max_row,1],vif_max,'\n\n')
        flush.console()
      }
      
      in_dat<-in_dat[,!names(in_dat) %in% vif_vals[max_row,1]]
      
    }
    
    return(names(in_dat))
    
  }
  
}
str(nn.train)
a = nn.train[-c(2,36,3,5,8,9,10,12,16,18,22,23,27)]#selecting all integer variables
str(a)
keep_dat = vif_func(a,thresh = 5,trace = T)
#library(car)
#vif(a)
vif_func(a[,-8],thresh = 5,trace = T)
#Only Job level is removed since it has high collinearity with age and monthly income etc..

##Converting categorical variables to dummy variables
ot.matrix <- model.matrix(~ OverTime - 1, data = nn.train)
nn.train <- data.frame(nn.train, ot.matrix)

Gender.matrix <- model.matrix(~ Gender - 1, data = nn.train)
nn.train <- data.frame(nn.train, Gender.matrix)

Dep.matrix <- model.matrix(~Department - 1, data = nn.train)
nn.train <- data.frame(nn.train,Dep.matrix)

Marital.matrix <- model.matrix(~ MaritalStatus - 1, data = nn.train)
nn.train <- data.frame(nn.train,Marital.matrix)

BT.matrix <- model.matrix(~ BusinessTravel - 1, data = nn.train)
nn.train <- data.frame(nn.train, BT.matrix)

EF.matrix <- model.matrix(~EducationField - 1, data = nn.train)
nn.train <- data.frame(nn.train,EF.matrix)

Job.matrix <- model.matrix(~ JobRole - 1, data = nn.train)
nn.train <- data.frame(nn.train, Job.matrix)

#scaling test data
ot.matrix <- model.matrix(~ OverTime - 1, data = nn.test)
nn.test <- data.frame(nn.test, ot.matrix)

Gender.matrix <- model.matrix(~ Gender - 1, data = nn.test)
nn.test <- data.frame(nn.test, Gender.matrix)

Dep.matrix <- model.matrix(~Department - 1, data = nn.test)
nn.test <- data.frame(nn.test,Dep.matrix)

Marital.matrix <- model.matrix(~ MaritalStatus - 1, data = nn.test)
nn.test <- data.frame(nn.test,Marital.matrix)

BT.matrix <- model.matrix(~ BusinessTravel - 1, data = nn.test)
nn.test <- data.frame(nn.test, BT.matrix)

EF.matrix <- model.matrix(~EducationField - 1, data = nn.test)
nn.test <- data.frame(nn.test,EF.matrix)

Job.matrix <- model.matrix(~ JobRole - 1, data = nn.test)
nn.test <- data.frame(nn.test, Job.matrix)

str(nn.train)
nn.train1 = nn.train[-c(3,5,8,12,16,18,22,23)]
nn.train1 = nn.train1[,-c(6,7,19)]#removing all the factor variables

colnames(nn.train1)
nn.train1[,-c(2,28,19,7)]##removing the response variables and unnecessary variables
keep.dat = vif_func(nn.train1[-c(2,28,19,7)],thresh = 5, trace = T)
#out of 56 variables, we have 42 left which we can pass to neural net

#scaling variables
x <- subset(nn.train1, select = c("Age","PercentSalaryHike","DailyRate","MonthlyRate", "MonthlyIncome","HourlyRate"))
x.scaled <- scale(x)
colnames(df)
nn.train2 = nn.train1[,-c(1,3,7,11,12,14)]
nn.train2 = cbind(nn.train2,x.scaled)
nn.train2 = nn.train2[complete.cases(nn.train2),]
nn.train2$Attrition_class=0
nn.train2$Attrition_class[nn.train2$Attrition=="Yes"]=1
#Building neural net
library(neuralnet)
paste(keep.dat,collapse='+')
nn1 <- neuralnet(formula = Attrition_class ~  Age+DailyRate+DistanceFromHome+Education+
                   +EnvironmentSatisfaction+HourlyRate+JobInvolvement+JobSatisfaction+
                   MonthlyRate+NumCompaniesWorked+PercentSalaryHike+PerformanceRating+
                   RelationshipSatisfaction+StockOptionLevel+TotalWorkingYears+TrainingTimesLastYear+
                   WorkLifeBalance+YearsAtCompany+YearsInCurrentRole+YearsSinceLastPromotion+
                   YearsWithCurrManager+OverTimeYes+GenderMale+MaritalStatusMarried+
                   MaritalStatusSingle+BusinessTravelTravel_Frequently+BusinessTravelTravel_Rarely+
                   EducationFieldMarketing+EducationFieldMedical+EducationFieldOther+EducationFieldTechnical.Degree+
                   JobRoleHuman.Resources+JobRoleLaboratory.Technician+JobRoleManager+JobRoleManufacturing.Director+JobRoleResearch.Director+JobRoleResearch.Scientist+
                   JobRoleSales.Executive+JobRoleSales.Representative , 
                 data = nn.train2, 
                 hidden = c(10,5),
                 err.fct = "sse",
                 linear.output = FALSE,
                 lifesign = "full",
                 lifesign.step = 25,
                 threshold = 0.1,
                 stepmax = 5000
)
plot(nn1)
nn1$net.result
quantile(nn1$net.result[[1]], c(0,1,5,10,25,50,75,90,95,99,100)/100)

library(caret)
misClassTable = data.frame(Target = nn.train2$Target, Predict.score = nn1$net.result[[1]] )
misClassTable$Classification = ifelse(misClassTable$Predict.score>0.2,1,0)
confusionMatrix(misClassTable$Target, misClassTable$Classification)

## deciling
misClassTable$deciles <- decile(misClassTable$Predict.score)
library(data.table)
tmp_DT = data.table(misClassTable)
rank <- tmp_DT[, list(
  cnt = length(Target), 
  cnt_resp = sum(Target), 
  cnt_non_resp = sum(Target == 0)) , 
  by=deciles][order(-deciles)]
rank$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank$cum_resp <- cumsum(rank$cnt_resp)
rank$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank$ks <- abs(rank$cum_rel_resp - rank$cum_rel_non_resp);
library(scales)
rank$rrate <- percent(rank$rrate)
rank$cum_rel_resp <- percent(rank$cum_rel_resp)
rank$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank)
# ks = 0.78- captured in 1st 2 deciles
##Applying the model on test data after scaling it
nn.test1 = nn.test[-c(3,5,8,12,16,18,22,23)]#removing all the factor variables
nn.test1 = nn.test1[-c(6,7,19)]
x1 <- subset(nn.test1, select = c("Age","PercentSalaryHike","DailyRate","MonthlyRate", "MonthlyIncome","HourlyRate"))
x1.scaled <- scale(x1)
nn.test2 = nn.test1[,-c(1,3,7,11,12,14)]
nn.test2 = cbind(nn.test2,x1.scaled)
nn.test2<-nn.test2[complete.cases(nn.test2),]
paste(keep.dat,collapse = ",")
#selecting all the variables on which the neural net is built
nn.test3 = nn.test2[,c(1,48,50,2,3,4,53,5,7,51,8,49,9,10,11,12,13,14,15,16,17,18,21,23,28,29,31,32,35,36,37,38,40,41,42,43,44,45,46,47)]
compute.output = compute(nn1, nn.test3[,-1])
#pred = prediction(nn1,nn.test2[-c(1)])
nn.test3$Predict.score = compute.output$net.result
quantile(nn.test3$Predict.score, c(0,1,5,10,25,50,75,90,95,99,100)/100)
nn.test3$Predict.Class[nn.test3$Predict.score < 0.2] = "No"
nn.test3$Predict.Class[nn.test3$Predict.score >=0.2] = "Yes"
table(nn.test3$Attrition,nn.test3$Predict.Class)
nn.test3$deciles = decile(nn.test3$Predict.score)
nn.test4 = nn.test3
nn.test4$Target[nn.test4$Attrition=="Yes"]=1
nn.test4$Target[nn.test4$Attrition=="No"]=0
nn.test4 = nn.test4[,-c(1)]
tmp_DT = data.table(nn.test4)
rank_test<- tmp_DT[, list(
  cnt = length(Target), 
  cnt_resp = sum(Target), 
  cnt_non_resp = sum(Target == 0)) , 
  by=deciles][order(-deciles)]
##str(rank_test)
rank_test$rrate <- round (rank$cnt_resp / rank$cnt,2);
rank_test$cum_resp <- cumsum(rank$cnt_resp)
rank_test$cum_non_resp <- cumsum(rank$cnt_non_resp)
rank_test$cum_rel_resp <- round(rank$cum_resp / sum(rank$cnt_resp),2);
rank_test$cum_rel_non_resp <- round(rank$cum_non_resp / sum(rank$cnt_non_resp),2);
rank_test$ks <- (rank$cum_rel_resp) - (rank$cum_rel_non_resp);
library(scales)
rank_test$rrate <- percent(rank$rrate)
rank_test$cum_rel_resp <- percent(rank$cum_rel_resp)
rank_test$cum_rel_non_resp <- percent(rank$cum_rel_non_resp)

View(rank_test)