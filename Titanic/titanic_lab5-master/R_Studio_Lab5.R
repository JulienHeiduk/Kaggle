#Tutorial1
setwd("D:/Fall2015/DA/DataAnalytics-master/DataAnalytics-master/Week6")
 train <- read.csv("D:/Fall2015/DA/DataAnalytics-master/DataAnalytics-master/Week6/train.csv")
   View(train)
 test <- read.csv("D:/Fall2015/DA/DataAnalytics-master/DataAnalytics-master/Week6/test.csv")
   View(test)
 str(train)
str(test)
summary(train)

 summary(test)
  
table(train$Survived)

 prop.table(table(train$Survived))

test$Survived <- rep(0, 418)
 submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
 write.csv(submit, file = "theyallperish.csv", row.names = FALSE)
 
 #Tutorial2:
 summary(train$Sex)
 prop.table(table(train$Sex, train$Survived))
 prop.table(table(train$Sex, train$Survived),1)
 test$Survived <- 0
 test$Survived[test$Sex == 'female'] <- 1
 summary(train$Age)
 train$Child <- 0
 train$Child[train$Age < 18] <- 1
 aggregate(Survived ~ Child + Sex, data=train, FUN=sum)
 aggregate(Survived ~ Child + Sex, data=train, FUN=length)
 aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
 train$Fare2 <- '30+'
 train$Fare2[train$Fare < 30 & train$Fare >= 20] <- '20-30'
 train$Fare2[train$Fare < 20 & train$Fare >= 10] <- '10-20'
 train$Fare2[train$Fare < 10] <- '<10'
 aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
 test$Survived <- 0
 test$Survived[test$Sex == 'female'] <- 1
 test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0
 #Tutorial 3:
 library(rpart)
 fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
 plot(fit)
 text(fit)
 install.packages('rattle')
 install.packages('rpart.plot')
 install.packages('RColorBrewer')
 library(rattle)
 library(rpart.plot)
 library(RColorBrewer)
 fancyRpartPlot(fit)
 Prediction <- predict(fit, test, type = "class")
 submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
 write.csv(submit, file = "myfirstdtree.csv", row.names = FALSE)
 fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
              method="class", control=rpart.control(minsplit=2, cp=0))
 fancyRpartPlot(fit)
 fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train,
              method="class", control=rpart.control( your controls ))
 new.fit <- prp(fit,snip=TRUE)$obj
 fancyRpartPlot(new.fit)
 
 #Tuorial 4:
 test$Survived <- NA
 combi <- rbind(train, test)
 combi$Name <- as.character(combi$Name)
 combi$Name[1]
 strsplit(combi$Name[1], split='[,.]')
 strsplit(combi$Name[1], split='[,.]')[[1]]
 strsplit(combi$Name[1], split='[,.]')[[1]][2]
 combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
 combi$Title <- sub(' ', '', combi$Title)
 table(combi$Title)
 combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
 combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
 combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
 combi$Title <- factor(combi$Title)
 combi$FamilySize <- combi$SibSp + combi$Parch + 1
 combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
 combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
 combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
 table(combi$FamilyID)
 famIDs <- data.frame(table(combi$FamilyID))
 famIDs <- famIDs[famIDs$Freq <= 2,]
 combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
 combi$FamilyID <- factor(combi$FamilyID)
 train <- combi[1:891,]
 test <- combi[892:1309,]
 fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
                +              data=train, method="class")
 fancyRpartPlot(fit)
 
 # Now let's make a prediction and write a submission file
 Prediction <- predict(fit, test, type = "class")
 submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
 write.csv(submit, file = "engineeredfeaturestree.csv", row.names = FALSE)
 
 #Tutorial 5:
 
 install.packages('randomForest')
 install.packages('party')
 library(randomForest) 
 library(rpart)
 library(party)
 sample(1:10, replace = TRUE)
 summary(combi$Age)
 Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                 data=combi[!is.na(combi$Age),], method="anova")
 combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
 summary(combi)
 summary(combi$Embarked)
 which(combi$Embarked == '')
 combi$Embarked[c(62,830)] = "S"
 combi$Embarked <- factor(combi$Embarked)
 summary(combi$Fare)
 which(is.na(combi$Fare))
 combi$FamilyID2 <- combi$FamilyID
 combi$FamilyID2 <- as.character(combi$FamilyID2)
 combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
 combi$FamilyID2 <- factor(combi$FamilyID2)
 train <- combi[1:891,]
 test <- combi[892:1309,]
 set.seed(415)
 fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
                       +                         FamilyID2, data=train, importance=TRUE, ntree=2000)
 varImpPlot(fit)
 Prediction <- predict(fit, test)
 
 submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction, row.names=NULL)
 write.csv(submit, file = "firstforest.csv", row.names = FALSE)
 library(party)
 set.seed(415)
 fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
                data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
 Prediction <- predict(fit, test, OOB=TRUE, type = "response")
 submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
 write.csv(submit, file = "ciforest.csv", row.names = FALSE)
 
