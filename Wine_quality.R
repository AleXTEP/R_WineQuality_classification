#####################################################################################################
#                                                                                                   #
#                                       WINE QUALITY                                                #
#                                                                                                   #
#####################################################################################################



getwd(); 
setwd("C:/Users/Utilisateur/Desktop/....")
wine = read.csv("C:/Users/Utilisateur/Desktop/wine/wine.csv", sep = ";")
is.factor(wine$quality);wine$quality = as.factor(wine$quality); is.factor(wine$quality)


#####################################################################################################
#packages used

library(Hmisc); library(pastecs); library(psych); library(caret); library(kernlab);
library(AppliedPredictiveModeling); library(neuralnet); library(e1071);library(RSNNS)
library(randomForest);library(pROC); library(FactoMineR); library(caTools);library(corrplot);
library(animation); library(FSelector);library(RSNNS);library(rpart);library(partykit);





#####################################################################################################
#                                       EXPLORATION                                                 #         
#####################################################################################################




dim(wine); names(wine)
is.factor(wine$quality); wine$quality = as.factor(wine$quality); is.factor(wine$quality)
summary(wine);describe(wine);stat.desc(wine[, 1:11]) 

wine[!complete.cases(wine),]
wine =na.omit(wine)


#univariate 
description_of_a_variable   =   function(var, name=""){
  
  stat<<-describe(var);
  layout(matrix(c(1,2,3,3), 2, 2, byrow = TRUE))
  qqnorm(var);qqline(var); shapiro<<-shapiro.test(var);
  h<<-hist(var, breaks=30, col="azure2", xlab="", main=name, freq=FALSE);
  lines(density(var), col="darkgreen", lwd=2)
  bp<<- boxplot(var, range = 0.3, varwidth = TRUE, horizontal = TRUE, col = "azure2", main = name)
  
  
}
description_of_a_variable(wine[,11], name="Alcohol"); shapiro; stat;

layout(matrix(c(1)))



#barplot
ggplot(wine, aes(quality))+geom_bar(width=.7)+coord_flip()+
  geom_text(aes(y = (..count..),label =   ifelse((..count..)==0,"",scales::percent((..count..)/sum(..count..)))), stat="bin",colour="red")+
  xlab("QUALITY") + ylab("QUANTITY")+ggtitle("Wine quality reparti")


#correlation
correlation <- cor(wine[, 1:11])
corrplot(correlation, method = "number")


# correlation by quality
transparentTheme(trans = 1) ;featurePlot(x = wine[, 4:5], y = wine$quality, plot = "pairs", auto.key = list(columns = 3))
# kernel distribution by quality
featurePlot(x = wine[, 1:11],y = wine$quality,plot = "density",scales = list(x = list(relation="free"),
                                                                             y = list(relation="free")),adjust = 1.5, pch = "|",auto.key = list(columns = 3))





####################################           PCA         #############################################  

data= wine
acp = PCA(data, scale.unit=TRUE, ncp=5, quali.sup=12, graph=T) 
data= wine[-c(4746,2782)]
acp= PCA(data,  scale.unit=TRUE, ncp=5, quali.sup=12, graph=F)

colors=c((adjustcolor("moccasin", alpha=1)),(adjustcolor("navajowhite", alpha=1)),(adjustcolor("navajowhite1", alpha=1)),
         (adjustcolor("navajowhite2", alpha=1)),(adjustcolor("navajowhite3", alpha=1)), (adjustcolor("navajowhite4", 1)), 
         (adjustcolor("red", alpha=1)))

acp = PCA(data, scale.unit=TRUE, ncp=5, quali.sup=12, graph=F) 
plot(acp, axes=c(1, 2),  choix="ind", habillage=12, col.hab=colors, lwd = 1, cex =1, label="none")
plot(acp, axes=c(1, 2),  choix="var", habillage=12, lwd = 1, cex = .7)
quality= data.frame(acp$quali.sup$coord)
plot(quality$Dim.1, quality$Dim.2, main ="Quality factor map", col=colors, cex = 2, pch=16); abline(h = 0, col = "blue", lty = "dotted");abline(v = 0, col = "blue", lty = "dotted")

seccol=adjustcolor("navajowhite2", alpha=0.8);maincol=adjustcolor("black", alpha=1)
axes=data.frame(acp$ind$coord)[,c(1,2)];names(axes)[]=paste(c("ax1", "ax2"))
cos=data.frame(acp$ind$cos2)[,c(1,2)];names(cos)[]=paste(c("cos_ax1", "cos_ax2"));
cos$sum_cos = cos$cos_ax1+cos$cos_ax2

results =cbind(axes, cos, data[, 12]);
results=results[order(-results$sum_cos),]

graph=function(cosinus, quality, quality2=""){
  rank= quality-2;   colors=c(seccol, seccol,seccol, seccol, seccol, seccol, seccol);  colors[rank]=maincol
  xyplot(results[results$sum_cos>cosinus,]$ax2~results[results$sum_cos>cosinus,]$ax1,main = quality2, group=results[,6],  
         panel=function(...){panel.xyplot(..., cex=1,  col=colors,
                                          pch=16, alpha=0.7);panel.abline(v=0, h=0)})
}

par(mfrow=c(2,2))
graph(0.6, 5, "Quality 5")
graph(0.6, 7, "Quality 7")

dynamic= function(cos) { ani.record(reset=TRUE); for(i in 3:9){plot(graph(cos, i));ani.record() }}
dynamic(0.4);oopts = ani.options(interval=1);ani.replay()

#####################################################################################################
#####################################################################################################
#####################################################################################################






#####################################################################################################
#                                       CLASSIFICATION                                              #         
#####################################################################################################

selections = data.frame(symmetrical.uncertainty(quality~., wine)); selections$characteristic=row.names(selections)
selections = selections[order(-selections$attr_importance),]; selections$sum= cumsum(selections$attr_importance)
selections$percent=selections$attr_importance/max(selections$sum); selections

prediction=function(data, model){
  pred<<-predict(model, newdata=data); true <<- data[,12]; 
  a=postResample(as.factor(pred), as.factor(true))
  pred=as.numeric(pred); is.numeric(pred)
  result <<- list(table(pred, true),multiclass.roc(data$quality,pred), a, varImp(model, scale = FALSE), plot(varImp(model, scale = FALSE), top = 11))
  return(result)
  tab<-as.data.frame.matrix( table(pred, true));tab[8,]=0;sum=0;for (i in 1:7){tab[8, i]= tab[i, i]/colSums(tab)[i];
                                                                               sum=sum+tab[8, i]};sum/7
  plot=plot(varImp(model, scale = FALSE), top = 11)
}



####################################           SVM         #############################################


#1   applications with R#

train=sample(nrow(wine), nrow(wine)*0.7)
svm1=svm(quality~., data=wine[train,], kernel="polynomial",  gamma=0.1, cost=1, degree = 3); 
summary(svm1)
#tune.out=tune(svm, quality~., data=wine[train,], kernel="polynomial", ranges=list(cost=3,gamma=c(0.1) summary(tune.out)

test = wine[-train,]
prediction(test, svm1)
prediction(wine, svm1)


#2 application CARET#

fitControl <- trainControl( method = "cv", number =4)
svm3Grid_bis <-  expand.grid(sigma=c(0.1, 0.2,  0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5),  C=c(1,2, 3, 5 , 7,8 ,10,15,30)) #0.7,  2

svm3Radial <- train(quality ~ ., data =  wine[train,], method = "svmRadial",tuneGrid=svm3Grid_bis,
                    trControl = fitControl, verbose = FALSE); 
svm3_bis <- train(quality ~ ., data =  wine_bis[train,], method = "svmRadial",tuneGrid=svm3Grid_bis,
                  trControl = fitControl, verbose = FALSE); 
svm3_standart <- train(quality ~ ., data =  wine[train,], method = "svmRadial",tuneGrid=svm3Grid_bis,
                       trControl = fitControl, verbose = FALSE, preProc=c("center", "scale")); 
svm3_standart_bis <- train(quality ~ ., data =  wine_bis[train,], method = "svmRadial",tuneGrid=svm3Grid_bis,
                           trControl = fitControl, verbose = FALSE, preProc=c("center", "scale")); 


svm3Radial;plot(svm3Radial, main = list("SVMRadial initial data", cex = 2, col = "blue", font = 2)); 
svm3_bis;plot(svm3_bis,main = list("SVMRadial reduced data", cex = 2, col = "blue", font = 2))
svm3_standart; plot(svm3_standart, main = list("SVMRadial Standardized  data", cex = 2, col = "blue", font = 2))
svm3_standart_bis;plot(svm3_standart_bis, , main = list("SVMRadial Standardized reduced data", cex = 2, col = "blue", font = 2))

prediction(test,svm3Radial)
prediction(test, svm3_bis)
prediction(test,svm3_standart)
prediction(test,svm3_standart_bis)



############################           RANDOM FOREST           #######################################

fitControl <- trainControl( method = "repeatedcv", number =4, repeats = 10 )
rfGrid <-  expand.grid(mtry=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
rfGrid2 <-  expand.grid(mtry=c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

RF =  train(quality ~ ., data =  wine[train,],  method = 'rf', tuneGrid=rfGrid, 
            trControl = fitControl, verbose = FALSE); 
RF_standart <- train(quality ~ ., data =  wine[train,],  method = 'rf', tuneGrid=rfGrid, 
                     trControl = fitControl, verbose = FALSE, preProc=c("center", "scale")); 
RF_bis =  train(quality ~ ., data =  wine_bis[train,],  method = 'rf', tuneGrid=rfGrid2, 
                trControl = fitControl, verbose = FALSE); 
RF_standart_bis <- train(quality ~ ., data =  wine_bis[train,],  method = 'rf', tuneGrid=rfGrid2, 
                         trControl = fitControl, verbose = FALSE, preProc=c("center", "scale")); 



RF;plot(RF, main = list("Random forest initial data", cex = 2, col = "blue", font = 2))
RF_standart;plot(RF_standart,  main = list("Random forest Standardized data", cex = 2, col = "blue", font = 2) )
RF_bis;plot(RF_bis, main = list("Random forest reduced data", cex = 2, col = "blue", font = 2))
RF_standart_bis; plot(RF_standart_bis, main = list("Random forest Standardized reduced data", cex = 2, col = "blue", font = 2))

prediction(test,RF)
prediction(test, RF_standart)
prediction(test, RF_bis)
prediction(test,RF_standart_bis)



#package RandomForest 

RFF= randomForest(quality ~ ., data=wine[train,], ntree=2000, mtry=1);
RFF; plot(RFF)
round(importance(RFF), 2);
predictionRFF=predict(RFF, wine[-train,]);
table(predictionRFF,wine[-train,12])

CVrff= rfcv(trainx=wine[train,1:11], trainy=wine[train,12], cv.fold=5, scale="log", step=0.5,
            mtry=function(p) max(1, floor(sqrt(p))), recursive=FALSE)

with(CVrff, plot(n.var, error.cv, log="x", type="o", lwd=2))
############################                 NN                #######################################

fitControl <- trainControl( method = "repeatedcv", number =4, repeats = 10 )
nnGrid <-  expand.grid(size=c(1,2,3,4,5, 7, 8, 10), decay=c(0.0001,0.001, 0.01, 0.05,  0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1, 2, 3, 5) ) #5, 

NNl <- train(quality ~ ., data =  wine[train,],  method = 'nnet', tuneGrid=nnGrid, trControl = fitControl, verbose = FALSE); 
NN_standart <- train(quality ~ ., data =  wine[train,],  method = 'nnet', tuneGrid=nnGrid, trControl = fitControl, verbose = FALSE, preProc=c("center", "scale")); 
NN_bis <- train(quality ~ ., data =  wine_bis[train,],  method = 'nnet', tuneGrid=nnGrid, trControl = fitControl, verbose = FALSE); 
NN_standart_bis <- train(quality ~ ., data =  wine_bis[train,],  method = 'nnet', tuneGrid=nnGrid, trControl = fitControl, verbose = FALSE, preProc=c("center", "scale")); 


NNl;plot(NNl, main = list("NN initial data", cex = 2, col = "blue", font = 2))
NN_standart;plot(NN_standart, main = list("NN Standardized data", cex = 2, col = "blue", font = 2))
NN_bis;plot(NN_bis, main = list("NN reduced data", cex = 2, col = "blue", font = 2))
NN_standart_bis;plot(NN_standart_bis, main = list("NN Standardized reduceddata", cex = 2, col = "blue", font = 2) )

pred<-predict(NNl, newdata=test);
true <-test[,12]; table(pred, true)
pred=as.numeric(pred); is.numeric(pred)
multiclass.roc(test$quality,pred)
postResample(as.factor(pred), as.factor(true))
######################################################################################################