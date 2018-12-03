# Demand-Forecasting-XGBoost
Product level Demand Forecasting using XGBoost 

install.packages("xgboost")
install.packages("readr")
install.packages("caret")
install.packages("car")
install.packages("mlr")
install.packages('e1071', dependencies=TRUE)
install.packages("randomForest")
install.packages("h2o")
install.packages("ggthemes")
install.packages("DiagrammeR")
library(randomForest)
library(h2o)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(data.table)
library(mlr)
library(ggplot2)
library(DiagrammeR)
library(ggthemes)


#######sales_data###############
setwd("C:/Users/amittal/Desktop/hackathon")
input<- read.table("sales_data_final.csv", header = T, sep = ",")
setnames(input, "srsdvsnnbr", "div_no")
setDT(input)

input <- input[Retail_Price>0]
input <- input[List_sell_price>0]
#################################

#input[["wknbr"]] <- as.factor(input[["wknbr"]])
#str(input)
###########is_holiday table#######################
is_holiday <- read.table("holiday.csv", header = T, sep = ",")
input <- input[,trandt:=as.Date(format(as.Date(trandt), "%m/%d/%Y"),"%m/%d/%Y")]
setnames(is_holiday, "date","trandt")
setDT(is_holiday)
is_holiday <- is_holiday[,trandt:=as.Date(trandt,"%m/%d/%Y")]
colSums(is.na(input))
input <- na.omit(input)
input <- merge(input, is_holiday, by = c('trandt'), all.x=T)
input[,X:=ifelse(is.na(X),0,X)]
##################################################



#########aggregation based on weeknbr##############
input <- input[,list(price_weekly_avg=mean(Retail_Price),
                     original_price_weekly_avg=mean(List_sell_price), 
                     qty_weekly_sum=sum(Quantity),
                     is_holiday=sum(X)),by=c("div_no","skuid","locnnbr","wknbr")]
input[,is_holiday:=ifelse(is_holiday>0,1,0)]
#nrow(input[is_holiday==1])
#nrow(input[is_holiday==0])
################################################


colSums(is.na(input))

######weekly_inventory data##########
inventory_week<- read.table("weekly_inventory_input.csv", header = T, sep = ",")
setnames(inventory_week, c("SKU_ID","LOCN_NBR", "WK_NBR","TTL_UN_QT"), c("skuid","locnnbr","wknbr", "week_inventory"))
setDT(inventory_week)
colSums(is.na(input))
input <- merge(input, inventory_week, by = c('div_no','skuid','locnnbr','wknbr'), all.x = T)
input <- input[week_inventory >= 0]
######################################

setDT(input)

#########removing undesired columns
input <- input[,-c(13)]


######removing data for less than 30 counts
input[, ':='(Count = .N), by = c('div_no', 'skuid', 'item_nbr')]
input <- input[Count>=30]   ####item selling for not more than 30 weeks are being removed
#input <- input[, -c(12)]  ##do not remove the count column it's an important factor
############################################################



##############Adding some more columns/feature engineering###################
input <- input[,discount := (original_price_weekly_avg - price_weekly_avg)/original_price_weekly_avg]
date_features <- read.table("date_features.csv", header = T, sep = ",")
setDT(date_features)
colSums(is.na(input))
input <- merge(input, date_features, by = c('wknbr'), all.x = T)
#write.csv(input, "sales_data_week.csv", row.names = F)
setkey(input, wknbr) ##sorting based on week number
summary(input)
str(input)
#############################################################################


###############converting numeric data to factors######################
str(input)
shouldBeCategorical <- c('skuid', 'locnnbr', 'wknbr', 'is_holiday', 'item_nbr', 'class_id', 'line_id', 'subline_id','yr','wk_indicator',
                         'quarter','Month')
for(v in shouldBeCategorical) {
  input[[v]] <- as.factor(input[[v]])
}
str(input)
#######################################################################

######segreagating based on different inputs
input1 <- input[div_no == 57]
input2 <- input[div_no == 77]
########################################################

######graph of qty_weekly_sum with skuid#### 
qplot(input2$skuid, input2$qty_weekly_sum, main = "With Outliers")



#########removing the outliers#############
#summary(input)
#input <- input[qty_weekly_sum>=0]
#q1<-quantile(input$units, probs=0.25)
#q3<- quantile(input$units, probs = 0.75)
#iqr <- 1.5*(q3-q1)
#iqr
#q3

#input <- input[units<2407 & units>0]
#qplot(input$wk_no, input$units, main = "Without Outliers")
#nrow(input) ##32254
###########################################



########making units normal(using log of units instead of units)#######



qplot(qty_weekly_sum, data = input2, bins = 50, main = "Left skewed distribution")

#input$qty_weekly_sum <- log(input$qty_weekly_sum+1)

#qplot(qty_weekly_sum, data = input, bins = 50, main = "Normal distribution after log transformation")



########################################################################



colSums(is.na(input2))
colSums(is.na(input1))
###if there are categorical features do one hot encoding#####


#################Preparing data for model training#####


setkey(input2, wknbr)
setkey(input1, wknbr)
train77 <- input2[ordinal_week<99]
test77 <- input2[ordinal_week>=99]
train57 <- input1[ordinal_week<99]
test57 <- input1[ordinal_week>=99]

setDT(train77)
setDT(test77)
setDT(train57)
setDT(test57)


##one hot encoding##########
new_tr77 <- model.matrix(~.+0,data = train77[,-c("qty_weekly_sum"),with=F])
new_ts77 <- model.matrix(~.+0,data = test77[,-c("qty_weekly_sum"),with=F])
new_tr57 <- model.matrix(~.+0,data = train57[,-c("qty_weekly_sum"),with=F])
new_ts57 <- model.matrix(~.+0,data = test57[,-c("qty_weekly_sum"),with=F])
############################




################matrix conversion####################
labels57 <- train57$qty_weekly_sum
ts_label57 <- test57$qty_weekly_sum
labels77 <- train77$qty_weekly_sum
ts_label77 <- test77$qty_weekly_sum


#train_157 <- as.matrix(train57[,-c("qty_weekly_sum")])
#test_157 <- as.matrix(test57[,-c("qty_weekly_sum")])
#train_177 <- as.matrix(train77[,-c("qty_weekly_sum")])
#test_177 <- as.matrix(test77[,-c("qty_weekly_sum")])


dtrain57 <- xgb.DMatrix(data = new_tr57,label = labels57)
dtest57 <- xgb.DMatrix(data = new_ts57,label=ts_label57)
dtrain77 <- xgb.DMatrix(data = new_tr77,label=labels77)
dtest77 <- xgb.DMatrix(data = new_ts77,label=ts_label77)
##################################################



###xgb.cv to get nround and eta and importance matrix########

set.seed(123)
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)
xgbcv57 <- xgb.cv( params = params, data = dtrain57, nrounds = 200, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)
xgbcv77 <- xgb.cv( params = params, data = dtrain77, nrounds = 200, nfold = 5, showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

#first default - model training
xgb157 <- xgb.train (params = params, data = dtrain57, nrounds = 80, 
                     watchlist = list(val=dtest57,train=dtrain57), print.every.n = 10, 
                     early.stop.round = 10, maximize = F , eval_metric = "rmse")


xgb177 <- xgb.train (params = params, data = dtrain77, nrounds = 15, 
                     watchlist = list(val=dtest77,train=dtrain77), print.every.n = 10, 
                     early.stop.round = 10, maximize = F , eval_metric = "rmse")

#########feature importance curve#############
mat57 <- xgb.importance (feature_names = colnames(new_tr57),model = xgb157)
xgb.plot.importance (importance_matrix = mat57[1:20])

mat77 <- xgb.importance (feature_names = colnames(new_tr77),model = xgb177)
xgb.plot.importance (importance_matrix = mat77[1:20])

##############################################




#####parameter tuning using mlr package#####################
#convert characters to factors
fact_col <- colnames(train57)[sapply(train57,is.character)]


for(i in fact_col) set(train57,j=i,value = factor(train[[i]]))
for (i in fact_col) set(test57,j=i,value = factor(test[[i]]))
for(i in fact_col) set(train77,j=i,value = factor(train[[i]]))
for (i in fact_col) set(test77,j=i,value = factor(test[[i]]))



#1. create a task both for train and test
traintask57 <- makeRegrTask(data = train57, target = "qty_weekly_sum")
testtask57 <- makeRegrTask(data = test57, target = "qty_weekly_sum")
traintask77 <- makeRegrTask(data = train77, target = "qty_weekly_sum")
testtask77 <- makeRegrTask(data = test77, target = "qty_weekly_sum")



#do one hot encoding`<br/> 
traintask57 <- createDummyFeatures (obj = traintask57) 
testtask57 <- createDummyFeatures (obj = testtask57)
traintask77 <- createDummyFeatures (obj = traintask77) 
testtask77 <- createDummyFeatures (obj = testtask77)


#2. make learner:
lrn57 <- makeLearner("regr.xgboost")
lrn57$par.vals <- list( objective="reg:linear", eval_metric="rmse", nrounds=80, eta=0.2)
lrn77 <- makeLearner("regr.xgboost")
lrn77$par.vals <- list( objective="reg:linear", eval_metric="rmse", nrounds=22, eta=0.2)

#3. Make parameter space:
params <- makeParamSet( makeDiscreteParam("booster",values = c("gbtree","gblinear")),
                        makeIntegerParam("max_depth",lower = 3,upper = 10),
                        makeIntegerParam("gamma",lower = 0,upper = 10),
                        makeNumericParam("lambda",lower = 0.55,upper = 0.60),
                        makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                        makeNumericParam("subsample",lower = 0.5,upper = 1), 
                        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#4. set resampling strategy
set_cv <- makeResampleDesc("CV",iters=5L)


#5. Setting search strategy:
rancontrol <- makeTuneControlRandom(maxit = 10L)   ###build 10 models


#6. Tune parameters:
xg_tune57 <- tuneParams(learner = lrn57, task = traintask57, resampling = set_cv, 
                     measures = mae, par.set = params, control = rancontrol, show.info = T)##for regression here I've taken mae(mean absolute error) for classification there are various eg. acc(accuracy)
xg_tune77 <- tuneParams(learner = lrn77, task = traintask77, resampling = set_cv, 
                      measures = mae, par.set = params, control = rancontrol, show.info = T)

xg_tune57$x
xg_tune77$x


#7. Set hyperparameters:
#lrn_new57 <- setHyperPars(learner = lrn57, par.vals = xg_tune57$x)
#lrn_new77 <- setHyperPars(learner = lrn77, par.vals = xg_tune77$x)


#8. train model:
#xgmodel57 <- train(learner = lrn_new57,task = traintask57)
#xgmodel77 <- train(learner = lrn_new77,task = traintask77)


#9. #predict model
#xgpred <- predict(xgmodel,testtask)
#postResample(test$qty_weekly_sum, xgpred)
#################################################################





############putting parameters obtained from mlr into xgboost##################
set.seed(123)
xgbFit57 = xgboost(booster = "gbtree", data = dtrain57, nfold = 5, nrounds = 80, objective = "reg:linear", eval_metric = "rmse", nthread = 8, eta = 0.2, gamma = 1, lambda = 0.568, max_depth = 10, min_child_weight = 2.54, subsample = 0.874, colsample_bytree = 0.947)
set.seed(1234)
xgbFit77 = xgboost(booster = "gbtree", data = dtrain77, nfold = 5, nrounds = 22, objective = "reg:linear", eval_metric = "rmse", nthread = 8, eta = 0.2, gamma = 0, lambda = 0.563, max_depth = 9, min_child_weight = 2.46, subsample = 0.537, colsample_bytree = 0.736)

#preds57_train <- predict(xgbFit57, newdata = as.matrix(train_157))
#preds77_train <- predict(xgbFit77, newdata = as.matrix(train_177))
#train57 <- train57[,predictions:= preds57_train]

preds57 <- predict(xgbFit57, newdata = dtest57)
preds77 <- predict(xgbFit77, newdata = dtest77)
test57 <- test57[,predictions:= preds57]
test77 <- test77[,predictions:= preds77]

write.csv(test57, "test57.csv", row.names = F)
write.csv(test77, "test77.csv", row.names = F)

postResample(test57$qty_weekly_sum, preds57)
postResample(test77$qty_weekly_sum, preds77)

##########div_item_sku level for ploit visualization########
#train57 <- train57[,div_item_sku:=paste0(div_no,"_",item_nbr,"_",skuid)]
#train77 <- train77[,div_itme_sku:=paste0(div_no,"_",item_nbr,"_",skuid)]

# Visualize
ggplot2::ggplot() + 
  # Training data points
  ggplot2::geom_point(data = test57, size = 2, aplha = 0.7,
                      ggplot2::aes(x = test57$wknbr, y = test57$qty_weekly_sum, color = "Training data")) +
  # XGBoost predictions
  ggplot2::geom_line(data = test57, size = 2, alpha = 0.7,
                     ggplot2::aes(x = test57$wknbr, y = test57$predictions,
                                  color = "Predicted with xgboost")) +
  # Linear regression predictions
  #ggplot2::geom_line(data = test_data, size = 2, alpha = 0.7,
                     #ggplot2::aes(x = x, y = y_predicted_linreg,
                      #            color = "Predicted with linear regression")) +
  # Hide legend title, change legend location and add axis labels
  ggplot2::theme(legend.title = element_blank(),
                 legend.position = "bottom") + labs(y = "Sold_Units",
                                                    x = "Week Number") +
  ggthemes::scale_colour_colorblind()



################### View the trees from a model
xgb.plot.tree(model = xgbFit57)

# View only the first tree in the XGBoost model
xgb.plot.tree(model = xgbFit57, n_first_tree = 0)
###############################################################################
