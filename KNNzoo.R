library(readr)
library(class)
zoo<-read.csv(file.choose())
zoo<-data.frame(zoo)
View(zoo)

names(zoo)<-c("animal","hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","size","type")
types<-table(zoo$type)
zoo_target<-zoo[,18]
zoo_key<-zoo[,1]
zoo$animal<-NULL
names(types)<-c("mammal","bird","reptile","fish","amphibian","insect","crustacean")

prediction<-knn.cv(zoo,zoo_target,k=(sqrt(17)+1),prob=TRUE)
data.frame(types)
table1<-table(zoo_target,prediction)
table1
sum(diag(table1))/sum(table1)*100
