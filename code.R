library(data.table)
library(h2o)
library(stringr)
library(caret)
library(randomForest)
library(plyr)
library(e1071)
train=fread('/home/gamechanger/Documents/hackerearth/p1/train.csv',na.strings = c(""," ","NA"),colClasses =c(verification_status_joint='character'))
test=fread('/home/gamechanger/Documents/hackerearth/p1/test.csv',na.strings = c("","NA"," "),colClasses =c(verification_status_joint='character'))

##for checking corelation
num_col <- colnames(train)[sapply(train, is.numeric)]
num_col=num_col[!(num_col%in%c("loan_status","member_id"))]
corrplot(cor(train[,ncol]),method="number")

##converting values to integer
train[,term := unlist(str_extract_all(string = term,pattern = "\\d+"))]
test[,term := unlist(str_extract_all(string = term,pattern = "\\d+"))]

train[,term := as.integer(term)]
test[,term := as.integer(term)]

##extracting character to number by using function of string lib,str_extract_all
train=train[emp_length == "n/a", emp_length := '-1']
train=train[emp_length == "< 1 year", emp_length := '0']
train=train[,emp_length := unlist(str_extract_all(emp_length,pattern = "\\d+"))]
train=train[,emp_length := as.integer(emp_length)]

test=test[emp_length=="n/a",emp_length:='-1']
test=test[emp_length=="< 1 year",emp_length:='0']
test=test[,emp_length:=unlist(str_extract_all(emp_length,pattern="\\d+"))]
test=test[,emp_length:=as.integer(emp_length)]

#convert initial_list_status
train=train[,initial_list_status:=as.integer(as.factor(initial_list_status))-1]
test=test[,initial_list_status:=as.integer(as.factor(initial_list_status))-1]
##fixing skewness in variables by using ggplot density graph

train=train[is.na(annual_inc),annual_inc:=0]
train=train[,annual_inc:=log(annual_inc+10)]
test=test[is.na(annual_inc),annual_inc:=0]
test=test[,annual_inc:=log(annual_inc+10)]

##checking skewness in other variables
num_col <- colnames(train)[sapply(train, is.numeric)]
num_col=num_col[!(num_col%in%c("loan_status","member_id"))]

skew=sapply(train[,num_col,with=F],function(x) skewness(x,na.rm=T))
skew=skew[skew>2]
train=train[,(names(skew)):=lapply(.SD,function(x) log(x+10)),.SDcols=names(skew)]

skew_t=sapply(train[,num_col,with=F],function(x) skewness(x,na.rm=T))
skew_t=skew_t[skew_t>2]
test=test[,names(skew):=lapply(.SD,function(x) log(x+10)),.SDcols=names(skew)]
  
train=train[,dti:=log(dti+10)]
test=test[,dti:=log(dti+10)]
train[,desc:=NULL]
test=test[,desc:=NULL]

train=train[,pymnt_plan:=NULL]
test=test[,pymnt_plan:=NULL]
train=train[,verification_status_joint:=NULL]
test=test[,verification_status_joint:=NULL]
train=train[,title:=NULL]
test=test[,title:=NULL]
train=train[,application_type:=NULL]
test=test[,application_type:=NULL]
train=train[,batch_enrolled:=NULL]
test=test[,batch_enrolled:=NULL]


#one-hot encoding
train_mod=train[,.(last_week_pay,sub_grade,grade,purpose,
                   verification_status,home_ownership)]
test_mod=test[,.(last_week_pay,sub_grade,grade,purpose,
                   verification_status,home_ownership)]
train_mod[is.na(train_mod)]='-1'
test_mod[is.na(test_mod)]='-1'

train_ex=model.matrix(~.+0,data=train_mod)
test_ex=model.matrix(~.+0,data=test_mod)
train_ex=as.data.table(train_ex)
test_ex=as.data.table(test_ex)
df=setdiff(colnames(train_ex),colnames(test_ex))
train_ex=train_ex[,-c(62,65,66,69,70,154)]

new_train=cbind(train,train_ex)
new_test=cbind(test,test_ex)
new_train=new_train[,c("last_week_pay","grade","sub_grade",
                       "purpose","verification_status",
                       "home_ownership"):=NULL]
new_test=new_test[,c("last_week_pay","grade","sub_grade",
                     "purpose","verification_status",
                     "home_ownership"):=NULL]

comb_data <- rbind(new_train,new_test,fill=TRUE)
#encoding for address state,zip_code,emp_title
for(i in colnames(comb_data)[sapply(comb_data,is.character)])
  set(x=comb_data,j=i,value=as.integer(as.factor(comb_data[[i]])))
F_train=comb_data[!is.na(loan_status)]
F_test=comb_data[is.na(loan_status)]
F_train=F_train[,c("funded_amnt","funded_amnt_inv","collection_recovery_fee") := NULL]
F_test=F_test[,c("funded_amnt","funded_amnt_inv","collection_recovery_fee") := NULL]



###feature engineering
F_train=F_train[,new_var_2:=log(annual_inc/loan_amnt)]
F_test=F_test[,new_var_2:=log(annual_inc/loan_amnt)]
F_train=F_train[,new_var_3:=total_rec_int+total_rec_late_fee]
F_test=F_test[,new_var_3:=total_rec_int+total_rec_late_fee]
F_train=F_train[,new_var_4:=sqrt(loan_amnt*int_rate)]
F_test=F_test[,new_var_4:=sqrt(loan_amnt*int_rate)]

train[,new_var_5 := mean(loan_amnt),grade]
F_train[,new_var_5 := train$new_var_5]

xkm <- train[,mean(loan_amnt),grade]

test <- xkm[test, on="grade"]
F_test[,new_var_5 := test$V1]

F_train=na.roughfix(F_train)
F_test=na.roughfix(F_test)
##give program more ram for faster execution
h2o.init(nthreads = -1,max_mem_size = "10G")
h2o_train=as.h2o(F_train)
h2o_test=as.h2o(F_test)
h2o_train$loan_status=h2o.asfactor(h2o_train$loan_status)


