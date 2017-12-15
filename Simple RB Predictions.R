
rm(list = ls())

#Read stats from 2015 and scoring data from 2016
RBData2013 = read.csv("C:\\Users\\tm9045\\Desktop\\Stanford\\Fall2016\\R\\2013RBData.csv", stringsAsFactors=FALSE)
RBData2014 = read.csv("C:\\Users\\tm9045\\Desktop\\Stanford\\Fall2016\\R\\2014RBData.csv", stringsAsFactors=FALSE)
RBData2015 = read.csv("C:\\Users\\tm9045\\Desktop\\Stanford\\Fall2016\\R\\2015RBData.csv", stringsAsFactors=FALSE)

library(XML)
library(RCurl)
library(rvest)

specialNames = c("Chris Johnson", "David Johnson", "Andre Brown", "Doug Martin")

#Grabs player age from wikipedia
getAge = function(name){
  search = gsub(' ', "_", name)
  url = paste("https://en.wikipedia.org/wiki/", search, sep="")
  html = read_html(url)
  age = html_nodes(html, ".ForceAgeToShow")
  textAge = html_text(age)
  numOnly = as.integer(substr(textAge, 6, 7))
  gotIt = length(numOnly)
  if (gotIt == 0){
    url = paste("https://en.wikipedia.org/wiki/", search, "_(American_football)",sep="")
    if (is.element(name, specialNames)){
      url = paste("https://en.wikipedia.org/wiki/", search, "_(running_back)",sep="")
    }
    html = read_html(url)
    age = html_nodes(html, ".ForceAgeToShow")
    textAge = html_text(age)
    numOnly = as.integer(substr(textAge, 6, 7))
    gotIt = length(numOnly)
  }
  return(numOnly)
}

#count number of rows in datasets
numRows2013=nrow(RBData2013)
numRows2014=nrow(RBData2014)
numRows2015=nrow(RBData2015)

#Classify players into draft rounds based on rank
RBData2013$draft = 0
RBData2014$draft = 0
RBData2015$draft = 0

RBData2013$draft = as.integer(RBData2013$draft)
RBData2014$draft = as.integer(RBData2014$draft)
RBData2015$draft = as.integer(RBData2015$draft)

for (i in 1:numRows2013){
  if (1<=160){
    decRound = (i-1)/10+1
    round = floor(decRound)
    RBData2013$draft[i] = round
  } else {
    RBData2013$draft[i] = 17
  }
}
for (i in 1:numRows2014){
  if (1<=160){
    decRound = (i-1)/10+1
    round = floor(decRound)
    RBData2014$draft[i] = round
  } else {
    RBData2014$draft[i] = 17
  }
}
for (i in 1:numRows2015){
  if (1<=160){
    decRound = (i-1)/10+1
    round = floor(decRound)
    RBData2015$draft[i] = round
  } else {
    RBData2015$draft[i] = 17
  }
}
for (i in 1:numRows2016){
  if (1<=160){
    decRound = (i-1)/10+1
    round = floor(decRound)
    RBData2016$draft[i] = round
  } else {
    RBData2016$draft[i] = 17
  }
}

RBData2013$draft = as.factor(RBData2013$draft)
RBData2014$draft = as.factor(RBData2014$draft)
RBData2015$draft = as.factor(RBData2015$draft)
RBData2016$draft = as.factor(RBData2016$draft)

#Add efficiency statistics
RBData2013$YDperCAR = RBData2013$YDS / RBData2013$RUSH
RBData2013$TYDS = RBData2013$YDS + RBData2013$ReYds
RBData2013$TDs = RBData2013$RuTD + RBData2013$ReTD
RBData2013$Touches = RBData2013$RUSH + RBData2013$REC
RBData2013$PPG = RBData2013$PTS / RBData2013$GMS

RBData2014$YDperCAR = RBData2014$YDS / RBData2014$RUSH
RBData2014$TYDS = RBData2014$YDS + RBData2014$ReYds
RBData2014$TDs = RBData2014$RuTD + RBData2014$ReTD
RBData2014$Touches = RBData2014$RUSH + RBData2014$REC
RBData2014$PPG = RBData2014$PTS / RBData2014$GMS

RBData2015$YDperCAR = RBData2015$YDS / RBData2015$RUSH
RBData2015$TYDS = RBData2015$YDS + RBData2015$ReYds
RBData2015$TDs = RBData2015$RuTD + RBData2015$ReTD
RBData2015$Touches = RBData2015$RUSH + RBData2015$REC
RBData2015$PPG = RBData2015$PTS / RBData2015$GMS

#add columns to data corresponding to player's score and draft rank next year
RBData2013$scoreNext = 0
RBData2013$draftNext = 0
RBData2013$ppgImprove = 0

RBData2014$scoreNext = 0
RBData2014$draftNext = 0
RBData2014$ppgImprove = 0

RBData2015$scoreNext = 0
RBData2015$draftNext = 0
RBData2015$ppgImprove = 0

#fill in 2016 score column in 2015 data
for (i in 1:numRows2013){
  if(RBData2013$Player[i] != "Giovani Bernard"){
    RBData2013$age[i] = getAge(RBData2013$Player[i]) - 3
  } else{
    RBData2013$age[i] = 24 - 3
  }
  for (j in 1:numRows2014){
    if (RBData2013$Player[i] == RBData2014$Player[j]) {
      RBData2013$scoreNext[i] = RBData2014$PTS[j]
      RBData2013$draftNext[i] = RBData2014$draft[j]
      RBData2013$ppgNext[i] = RBData2014$PPG[j]
      if (RBData2014$PPG[j]>RBData2013$PPG[i]){
        RBData2013$ppgImprove[i] = 1
      }
    }
  }
}
for (i in 1:numRows2014){
  if(RBData2014$Player[i] != "Giovani Bernard"){
    print(RBData2014$Player[i])
    RBData2014$age[i] = getAge(RBData2014$Player[i]) - 2
  } else{
    RBData2014$age[i] = 24 - 2
  }
  for (j in 1:numRows2015){
    if (RBData2014$Player[i] == RBData2015$Player[j]) {
      RBData2014$scoreNext[i] = RBData2015$PTS[j]
      RBData2014$draftNext[i] = RBData2015$draft[j]
      RBData2014$ppgNext[i] = RBData2015$PPG[j]
      if (RBData2015$PPG[j]>RBData2014$PPG[i]){
        RBData2014$ppgImprove[i] = 1
      }
    }
  }
}

for (i in 1:numRows2013){
  if (RBData2013$age[i]<=24){
    RBData2013$ageClass[i]="Young"
  } 
  else if (RBData2013$age[i]>=28){
    RBData2013$ageClass[i]="Old"
  } 
  else{
    RBData2013$ageClass[i]="Normal"
  } 
}

for (i in 1:numRows2014){
  if (RBData2014$age[i]<=24){
    RBData2014$ageClass[i]="Young"
  } 
  else if (RBData2014$age[i]>=28){
    RBData2014$ageClass[i]="Old"
  } 
  else{
    RBData2014$ageClass[i]="Normal"
  } 
}

RBData2013$ageClass = as.factor(RBData2013$ageClass)
RBData2014$ageClass = as.factor(RBData2014$ageClass)

RBData2013$draftNext = as.factor(RBData2013$draftNext)
RBData2014$draftNext = as.factor(RBData2014$draftNext)

RBData2013$ppgImprove = as.factor(RBData2013$ppgImprove)
RBData2014$ppgImprove = as.factor(RBData2014$ppgImprove)

#make new dataframes only including players who have scores in consecutive years
RBData2013relevant = RBData2013[1,]
for (i in 2:numRows2013){
  if (RBData2013$scoreNext[i] != 0){
    RBData2013relevant = rbind(RBData2013relevant,RBData2013[c(i),])
  }
}

RBData2014relevant = RBData2014[1,]
for (i in 2:numRows2014){
  if (RBData2014$scoreNext[i] != 0){
    RBData2014relevant = rbind(RBData2014relevant,RBData2014[c(i),])
  }
}

#Use randomforrest to see which factors most affect draft position
library(randomForest)
set.seed(1)
library(party)

draft.rf = randomForest(draftNext ~ PTS + YDperCAR + TYDS + Touches + TDs + PPG + ageClass, data=droplevels(RBData2013relevant), importance=TRUE, proximity=TRUE)
importance(draft.rf)
plot(draft.rf, log="y")
print(draft.rf)
varImpPlot(draft.rf)
ct = ctree(draftNext ~ PTS + YDperCAR + TYDS + Touches + TDs + PPG + ageClass, data = RBData2013relevant)
plot(ct, main="Conditional Inference Tree")
prediction = predict(draft.rf, RBData2014relevant)
t = table(prediction, RBData2014relevant$draftNext)
print(t)


improve.rf = randomForest(ppgImprove ~ PTS + YDperCAR + TYDS + Touches + TDs + PPG + ageClass, data=droplevels(RBData2013relevant), importance=TRUE, proximity=TRUE)
importance(improve.rf)
plot(improve.rf, log="y")
print(improve.rf)
varImpPlot(improve.rf)
ct = ctree(ppgImprove ~ PTS + YDperCAR + TYDS + Touches + TDs + PPG + ageClass, data = RBData2013relevant)
plot(ct, main="Conditional Inference Tree")
prediction = predict(improve.rf, RBData2014relevant)
t = table(prediction, RBData2014relevant$ppgImprove)
print(t)

#% chance that each player will improve or not the next year
impFit2013 = glm(ppgImprove ~ PTS + YDperCAR + TYDS + Touches + TDs + PPG + ageClass, data = RBData2013relevant, family = "binomial")
summary(impFit2013)
coef(impFit2013)
RBData2013relevant$ppgPredict= predict(impFit2013, newdata = RBData2013relevant, type = "response")

numCorrect2013=0
for (i in 1:nrow(RBData2013relevant)){
  if (round(RBData2013relevant$ppgPredict[i])==RBData2013relevant$ppgImprove[i]){
    numCorrect2013 = numCorrect2013+1
  }
}
accuracy2013 = numCorrect2013/nrow(RBData2013relevant)
accuracy2013

RBData2014relevant$ppgPredict= predict(impFit2013, newdata = RBData2014relevant, type = "response")

numCorrect2014=0
for (i in 1:nrow(RBData2014relevant)){
  if (round(RBData2014relevant$ppgPredict[i])==RBData2014relevant$ppgImprove[i]){
    numCorrect2014 = numCorrect2014+1
  }
}
accuracy2014 = numCorrect2014/nrow(RBData2014relevant)
accuracy2014

library(ggplot2)
ggplot(data = RBData2013relevant, aes(x = ppgPredict, y = ppgImprove)) + geom_point() + ggtitle("2013-2014 Without Age") + labs(x="Probibility of Improvement",y="Binary Improvement") 
ggplot(data = RBData2014relevant, aes(x = ppgPredict, y = ppgImprove)) + geom_point() + ggtitle("2014-2015 Without Age") + labs(x="Probibility of Improvement",y="Binary Improvement") 

