#Installing the libraries needed to run the code if not already installed

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
if(!require(DMwR)) install.packages("DMwR", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ROSE)
library(DMwR)
library(randomForest)
options(digits=3)

# Downloading the data set and reading the file

wine<-"https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
basename(wine)
download.file(wine,basename(wine))
winedata<-read.csv("winequality-red.csv")

# First look at file to check if we need to wrangle the data
head(winedata)

# Data wrangling, removing delimiters and renaming columns
winedata<-read_delim("winequality-red.csv", 
                    delim = ";", 
                    locale = locale(decimal_mark = ".", 
                                    grouping_mark = ","), 
                    col_names = TRUE)

cnames <- c("fixed_acidity", "volatile_acidity", "citric_acid",
            "residual_sugar", "chlorides", "free_sulfur_dioxide",
            "total_sulfur_dioxide", "density", "pH",
            "sulphates", "alcohol", "quality")

colnames(winedata)<-cnames
winedata<-as.data.frame(winedata)
head(winedata)
dim(winedata)

#Data exploration
#Wine quality histogram
winedata%>%ggplot(aes(quality))+geom_histogram(bins=20)

#Applying cutoff and factorizing on wine quality 
winedata<-winedata%>%mutate(quality=ifelse(quality>=7,"1","0"))%>%mutate(quality=factor(quality))

#New distribution of wine quality
histogram(winedata$quality)
table(winedata$quality)
prop.table(table(winedata$quality))

#Boxplots of features vs. wine quality
winedata%>%gather(wine_properties, values, -quality)%>%
  ggplot(aes(quality, values, fill=quality))+geom_boxplot()+
  facet_wrap(~wine_properties,scales="free")+
  theme(axis.text.x = element_blank())

#modeling
#data partition
set.seed(2020, sample.kind = "Rounding")
index<-createDataPartition(winedata$quality,times=1, p=0.2, list=FALSE)
test<-winedata[index,]
train<-winedata[-index,]

#Distribution of wine quality in the new datasets
table(train$quality)
table(test$quality)
prop.table(table(train$quality))
prop.table(table(test$quality))


#Logistic Regression model
glm_model<-train(quality~., method="glm", data=train)
glm_model_preds<-predict(glm_model,test)
confusionMatrix(data=glm_model_preds,reference=test$quality, positive = "1")

#K-Nearest Neighbours 
set.seed(2020,sample.kind = "Rounding")
tuning<-data.frame(k=seq(3,31,2))
knn_model<-train(quality~., method="knn", data=train,tuneGrid = tuning)
plot(knn_model)
knn_model$bestTune
knn_model_preds<-predict(knn_model,test)
confusionMatrix(data=knn_model_preds,reference=test$quality, positive="1")


#Random forest model
set.seed(2020,sample.kind = "Rounding")
tuning<-data.frame(mtry=c(1:5))
rf_model<-train(quality~.,method="rf",
                tuneGrid=tuning,
                importance=TRUE,
                data=train)
rf_model$bestTune
plot(rf_model)
rf_model_preds<-predict(rf_model,test)
confusionMatrix(data=rf_model_preds,reference=test$quality,positive="1")



#Upsampling
set.seed(2020,sample.kind = "Rounding")
train_up<-upSample(x=train[,-ncol(train)],y=train$quality)
table(train_up$Class)

#Downsampling
set.seed(2020,sample.kind = "Rounding")
train_down<-downSample(x=train[,-ncol(train)],y=train$quality)
table(train_down$Class)


#ROSE
library(ROSE)
set.seed(2020,sample.kind = "Rounding")
train_rose<-ROSE(quality ~ ., data=train)$data
table(train_rose$quality)

#SMOTE
library(DMwR)
set.seed(2020,sample.kind = "Rounding")
train_smote<-SMOTE(quality~., data=train)
table(train_smote$quality)



#Using random forest in upsampling
set.seed(2020,sample.kind = "Rounding")
tuning<-data.frame(mtry=c(1,2,3,4,5))
train_rf_up<-train(Class~.,method="rf",
                tuneGrid=tuning,
                importance=TRUE,
                data=train_up)
train_rf_up$bestTune
rf_preds_up<-predict(train_rf_up,test)
confusionMatrix(data=rf_preds_up,reference=test$quality, positive="1")
varImp(train_rf_up)
plot(train_rf_up)


#Using random forest in downsampling
set.seed(2020,sample.kind = "Rounding")
tuning<-data.frame(mtry=c(1,2,3,4,5))
train_rf_down<-train(Class~.,method="rf",
                   tuneGrid=tuning,
                   importance=TRUE,
                   data=train_down)
train_rf_down$bestTune
rf_preds_down<-predict(train_rf_down,test)
confusionMatrix(data=factor(rf_preds_down),
                reference=factor(test$quality),
                positive="1")
varImp(train_rf_down)
plot(train_rf_down)

#Using random forest in ROSE
set.seed(2020,sample.kind = "Rounding")
tuning<-data.frame(mtry=c(1,2,3,4,5))
train_rf_rose<-train(quality~.,method="rf",
                     tuneGrid=tuning,
                     importance=TRUE,
                     data=train_rose)
train_rf_rose$bestTune
rf_preds_rose<-predict(train_rf_rose,test)
confusionMatrix(data=factor(rf_preds_rose),
                reference=factor(test$quality),
                positive="1")
varImp(train_rf_rose)
plot(train_rf_rose)

#Using random forest in SMOTE
set.seed(2020,sample.kind = "Rounding")
tuning<-data.frame(mtry=c(1,2,3,4,5))
train_rf_smote<-train(quality~.,method="rf",
                     tuneGrid=tuning,
                     importance=TRUE,
                     data=train_smote)
train_rf_smote$bestTune
rf_preds_smote<-predict(train_rf_smote,test)
confusionMatrix(data=factor(rf_preds_smote),
                reference=factor(test$quality),
                positive="1")
varImp(train_rf_smote)
plot(train_rf_smote)


# Calculation of the F meas for all our Random Forest models, with beta=0.25
b<-0.25
F_meas_rf<-F_meas(rf_model_preds,test$quality, beta=b)
F_meas_rf_up<-F_meas(rf_preds_up,test$quality, beta=b)
F_meas_rf_down<-F_meas(rf_preds_down,test$quality, beta=b)
F_meas_rf_rose<-F_meas(rf_preds_rose,test$quality, beta=b)
F_meas_rf_smote<-F_meas(rf_preds_smote,test$quality, beta=b)

f_meas_values<-c(F_meas_rf,F_meas_rf_up,F_meas_rf_down,
                          F_meas_rf_rose,F_meas_rf_smote)
model<-c("Random Forest", "Random Forest Upsampling", "Random Forest Downsampling",
          "Random Forest ROSE", "Random Forest SMOTE")
data.frame(model,f_meas_values)%>%knitr::kable()

# Calculation of the F meas for all our Random Forest models, with beta=0.5
b<-0.5
F_meas_rf<-F_meas(rf_model_preds,test$quality, beta=b)
F_meas_rf_up<-F_meas(rf_preds_up,test$quality, beta=b)
F_meas_rf_down<-F_meas(rf_preds_down,test$quality, beta=b)
F_meas_rf_rose<-F_meas(rf_preds_rose,test$quality, beta=b)
F_meas_rf_smote<-F_meas(rf_preds_smote,test$quality, beta=b)

f_meas_values<-c(F_meas_rf,F_meas_rf_up,F_meas_rf_down,
                 F_meas_rf_rose,F_meas_rf_smote)
model<-c("Random Forest", "Random Forest Upsampling", "Random Forest Downsampling",
         "Random Forest ROSE", "Random Forest SMOTE")
data.frame(model,f_meas_values)%>%knitr::kable()


# Calculation of the F meas for all our Random Forest models, with beta=1
b<-1
F_meas_rf<-F_meas(rf_model_preds,test$quality, beta=b)
F_meas_rf_up<-F_meas(rf_preds_up,test$quality, beta=b)
F_meas_rf_down<-F_meas(rf_preds_down,test$quality, beta=b)
F_meas_rf_rose<-F_meas(rf_preds_rose,test$quality, beta=b)
F_meas_rf_smote<-F_meas(rf_preds_smote,test$quality, beta=b)

f_meas_values<-c(F_meas_rf,F_meas_rf_up,F_meas_rf_down,
                 F_meas_rf_rose,F_meas_rf_smote)
model<-c("Random Forest", "Random Forest Upsampling", "Random Forest Downsampling",
         "Random Forest ROSE", "Random Forest SMOTE")
data.frame(model,f_meas_values)%>%knitr::kable()

