library(readr)
library(class)
glass<-read.csv(file.choose())
glass<-data.frame(glass)
View(glass)

types<-table(glass$Type)
glass_target<-glass[,10]
glass_key<-glass[,1]
names(types)<-c("Building Window FP","Building Window NFP","Vehicle Window FP","Containers","Tableware","Headlamps")

prediction<-knn.cv(glass,glass_target,k=(sqrt(9)+1),prob=TRUE)
data.frame(types)
table1<-table(glass_target,prediction)
table1
sum(diag(table1))/sum(table1)*100

library(ggplot2)
plot(glass$Type,prediction)
