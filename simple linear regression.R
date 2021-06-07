#######################problem 1###########################
library(readxl)   #invoke library

# Load the data
data_cal_wt <- read.csv(file.choose(), header=TRUE)
View(data_cal_wt)
attach(data_cal_wt)

# Graphical exploration
plot(data_cal_wt)

dotplot(Weight.gained..grams., main = "Dot Plot of Waist Circumferences")
dotplot(Calories.Consumed, main = "Dot Plot of Adipose Tissue Areas")


boxplot(Weight.gained..grams., col = "dodgerblue4")
boxplot(Calories.Consumed, col = "red", horizontal = T)

hist(Weight.gained..grams.)
hist(Calories.Consumed)

# Normal QQ plot
qqnorm(Weight.gained..grams.)
qqline(Weight.gained..grams.)

qqnorm(Calories.Consumed)
qqline(Calories.Consumed)

hist(Weight.gained..grams., prob = TRUE)   # prob=TRUE for probabilities not counts
lines(density(Weight.gained..grams.))     # add a density estimate with defaults
lines(density(Weight.gained..grams., adjust = 2), lty = "dotted")   # add another "smoother" density

hist(Calories.Consumed, prob = TRUE)          # prob=TRUE for probabilities not counts
lines(density(Calories.Consumed))             # add a density estimate with defaults
lines(density(Calories.Consumed, adjust = 2), lty = "dotted")   # add another "smoother" density

# Bivariate analysis
# Scatter plot
plot(Weight.gained..grams., Calories.Consumed, main = "Scatter Plot", col = "Dodgerblue4", 
     col.main = "Dodgerblue4", col.lab = "Dodgerblue4", xlab = "Waist Ciscumference", 
     ylab = "Adipose Tissue area", pch = 20)  # plot(x,y)

# Exploratory data analysis
summary(data_cal_wt)

# Covariance
cov(Weight.gained..grams.,Calories.Consumed)

# Correlation Coefficient
cor(Weight.gained..grams.,Calories.Consumed)

#model building 
# Linear Regression model

reg <- lm(Weight.gained..grams.~ Calories.Consumed)
summary(reg)

confint(reg,leavel=0.95)

pred<-predict(reg,interval = "predict")
pred <- as.data.frame(pred) # changing into dataframe
View(pred)

# ggplot for adding Regression line for data
library(ggplot2)

ggplot(data = data_cal_wt, aes(x = Calories.Consumed, y = Weight.gained..grams.)) + 
  geom_point(color = 'blue') +
  geom_line(color = 'red', data = data_cal_wt, aes(x =Calories.Consumed, y = Weight.gained..grams.))

# Evaluation the model 

cor(pred$fit, Weight.gained..grams.)

rmse <- sqrt(mean(reg$residuals^2))
rmse


# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y

plot(log(Calories.Consumed),Weight.gained..grams.)
cor(log(Calories.Consumed), Weight.gained..grams.)

reg_log <- lm(Weight.gained..grams. ~ log(Calories.Consumed), data = data_cal_wt)
summary(reg_log)

confint(reg_log,level = 0.95)
pred <- predict(reg_log, interval = "predict")

pred <- as.data.frame(pred)
cor(pred$fit,Weight.gained..grams.)

rmse <- sqrt(mean(reg_log$residuals^2))
rmse


# Log transformation applied on 'y'
# input = x; output = log(y)

plot(Calories.Consumed, log(Weight.gained..grams.))
cor(Calories.Consumed, log(Weight.gained..grams.))

reg_log1 <- lm(log(Weight.gained..grams.) ~ Calories.Consumed, data = data_cal_wt)
summary(reg_log1)

predlog <- predict(reg_log1, interval = "predict")
predlog <- as.data.frame(predlog)
reg_log1$residuals
sqrt(mean(reg_log1$residuals^2))

pred <- exp(predlog)  # Antilog = Exponential function
pred <- as.data.frame(pred)

rmse <- sqrt(mean(res_log1^2))
rmse

# transform the variables to check whether the predicted values are better

reg_sqrt <- lm(Weight.gained..grams.~ sqrt(Calories.Consumed))
summary(reg_sqrt)

confint(reg_sqrt,level=0.95)

predict(reg_sqrt,interval="predict")

# transform the variables to check whether the predicted values are better
reg_log1 <- lm(Weight.gained..grams.~ log(Calories.Consumed))
summary(reg_log1)

confint(reg_log1,level=0.95)

predict(reg_log1,interval="predict")

# Result
## Applying transformation is decreasing Multiple R Squared Value. 
#So model doesnot need further transformation, Multiple R-squared:0.8968
######################problem 2##############################
# Loading the data
delivery_time<-read.csv(file.choose())
library(readr)
dt_st <-read.csv(file.choose())
View(dt_st) #view dataset
summary(dt_st) #summary of dataset/EDA

#Scatterplot of input Vs otput
plot(dt_st$`Sorting.Time`, dt_st$`Delivery.Time`)  # plot(X,Y)

#From plot we can say that data is linearity is there,strength is moderate (sub to check r value),
#& Direction is positive

#attached dataset
attach(dt_st)

#Correlation between output to input
cor(`Sorting.Time`, `Delivery.Time`) #cor(x,y)
#from value of correlation coe.(r) we can say that moderate correlation between o/p & i/p

# Simple Linear Regression model
reg <- lm(`Delivery.Time`~ `Sorting.Time`) # lm(Y ~ X)

#Summary of regression model
summary(reg)
#first thing is that variable is siginificant as value is less than 0.05
#R^2 value is less than 0.80 so we can say that model is underfit (moderately good).
#We can write eq. as DT=6.5227+1.6490(ST)

#check fitted values(predicted)
reg$fitted.values
reg$residuals

#but we have to check with predicted values
pred <- predict(reg)

#Check for error associated with each obs.
reg$residuals
sum(reg$residuals)

#check for mean of sum of errors is equal to 0.
mean(reg$residuals)
hist(reg$residuals) # check errors are normally distributed or not.

#Check for RMSE value
sqrt(sum(reg$residuals^2)/nrow(dt_st))  #RMSE
sqrt(mean(reg$residuals^2))

#interval for 5% of confidence
confint(reg,level=0.95)

predict(reg,interval="predict")

#visualising model
library(ggplot2)

ggplot(data = dt_st, aes(x = `Sorting.Time`, y = `Delivery.Time`)) + 
  geom_point(color='blue') +
  geom_line(color='red',data = dt_st, aes(x= `Sorting.Time`, y=pred))

#Inferences-
#From all above  value or correlatio coe.r is 0.82 which is moderateely acceptable ,
# function is linear in nature i.e. lm(DT ~ ST), 
# Coe. are significant and coe.of Determination value (R^2) is 0.682 which is also moderately acceptable
# mean of errors is 4.758099e-17 which is almost 0 ans errors are almost normally distributed.
# RMSE value is 2.79165 which is nearest to lower range value of delivery time
# so as model underfits,we need to go for transformation

# Logarithamic Model

# x = log(Sorting Time); y = Delivery Time

#Scatterplot of input Vs output
plot(log(`Sorting.Time`), `Delivery.Time`)   # plot(log(X),Y)
#From plot we can say that data is linearity is there,strength is moderate (sub to check r value),
#& Direction is positive

#Correlation between output to input
cor(log(`Sorting.Time`), `Delivery.Time`)

# Simple Linear Regression model-log transform
reg_log <- lm(`Delivery.Time` ~ log(`Sorting.Time`))   # lm(Y ~ log(X)

#Summary of regression model
summary(reg_log)
#first thing is that variable is significant as value is less than 0.05
#R^2 value is less than 0.80 so we can say that model is moderate and improved than previous
#We can write eq. as DT=1.160+9.043(log(ST))

#check fitted values(predicted)
reg_log$fitted.values
reg_log$residuals

#but we have to check with predicted values
predict(reg_log)

#Check for error associated with each obs.
reg_log$residuals
sum(reg_log$residuals)

#check for mean of sum of errors is equal to 0.
mean(reg_log$residuals)
hist(reg_log$residuals) # check errors are normally distributed or not.

#Check for RMSE value
sqrt(sum(reg_log$residuals^2)/nrow(dt_st))  #RMSE
sqrt(mean(reg_log$residuals^2))

#interval for 5% of confidence
confint(reg_log,level=0.95)
predict(reg_log,interval="confidence")

#visualing model
library(ggplot2)

ggplot(data = dt_st, aes(x = log(`Sorting.Time`), y = `Delivery.Time`)) + 
  geom_point(color='blue') +
  geom_line(color='red',data = dt_st, aes(x=log(`Sorting.Time`), y=pred))

#Inferences-
#From all above  value or correlation coe.(r) is 0.83 which is moderateely acceptable improved than previuos ,
# function is linear in nature i.e. lm(DT ~ log(ST)), 
# Coe. are significant and coe.of Determination value (R^2) is 0.695 which is also moderately acceptable and improved than previous.
# mean of errors is -1.863589e-16 which is almost 0 ans errors are almost normally distributed.
# RMSE value is 2.73 decreased slightly than previous model which is nearest to lower range value of delivery time
# so as model underfits,we need to go for another transformation to improve (r^2) value.

# Exponential Model
# x = Sorting Time and y = log(Delivery Time)

#Scatterplot of input Vs output
plot(`Sorting.Time`, log(`Delivery.Time`)) #plot(x,log(y))
#From plot we can say that data is linearity is there,strength is moderate (sub to check r value),
#& Direction is positive

#Correlation between output to input
cor(`Sorting.Time`, log(`Delivery.Time`)) #cor(x,log(y))
#from value of correlation coe.(r) we can say that very good correlation between o/p & i/p

# Simple Linear Regression model-exp transform
reg_exp <- lm(log(`Delivery.Time`) ~ `Sorting.Time` )  #lm(log(Y) ~ X)

#Summary of regression model
summary(reg_exp)
#first thing is that variable is siginificant as value is less than 0.05
#R^2 value is slight less than 0.80 best fit so moderate value  so we can say that model is somewhat bestfit as of now
#We can write eq. as log(DT)=2.12+0.1055(ST)

#check fitted values(predicted)
reg_exp$fitted.values
reg_exp$residuals

#but we have to check with predicted values
predict(reg_exp)

#convert exp values to normal
logdt <- predict(reg_exp)
dt <- exp(logdt)

#Check for error associated with each obs.
error = dt_st$`Delivery.Time` - dt
error
sum(error)

#check for mean of sum of errors is equal to 0.
mean(error)
hist(error) # check errors are normally distributed or not.

#Check for RMSE value
sqrt(sum(error^2)/nrow(dt_st))  #RMSE
sqrt(mean(error^2))

#interval for 5% of confidence
confint(reg_exp,level=0.95)
predict(reg_exp,interval="confidence")

#visualising model
library(ggplot2)

ggplot(data = dt_st, aes(x = `Sorting.Time`, y = log(`Delivery.Time`))) + 
  geom_point(color='blue') +
  geom_line(color='red',data = dt_st, aes(x=`Sorting.Time`, y=pred))

#Inferences-
#From all above  value or correlation coe.(r) is 0.843 which is good and improved than previuos ,
# function is linear in nature i.e. lm(log(DT) ~ ST), 
# Coe. are significant and coe.of Determination value (R^2) is 0.7109 which is also moderately acceptable and improved than previous.
# mean of errors is 0.1981094 which is almost 0 ans errors are normally distributed.
# RMSE value is 2.94025 increased slightly than previous model which is nearest to lower range value of delivery time
# so as model underfits,we need to go for another transformation to improve (r^2) value.


# Polynomial model with 2 degree (quadratic model)

# x = Sorting Time and y = `Sorting Time`+ I(`Sorting Time`*`Sorting Time`)

#Scatterplot of input Vs output
plot(`Sorting.Time`, `Delivery.Time`)
plot(`Sorting.Time`*`Sorting.Time`, `Delivery.Time`)
#From plot we can say that data is linearity is there,strength is moderate (sub to check r value),
#& Direction is positive

#Correlation between output to input
cor(`Sorting.Time`, `Delivery.Time`)
cor(`Sorting.Time`*`Sorting.Time`,(`Delivery.Time`))
#from value of correlation coe.(r) we can say that moderately acceptable correlation between o/p & i/p


# Simple Linear Regression model-polynomial with 2nd degree
#transform# lm(Y ~ X + I(X*X))

reg2degree <- lm((`Delivery.Time`) ~ `Sorting.Time` + I(`Delivery.Time`*`Delivery.Time`))

#Summary of regression model
summary(reg2degree)
#first thing is that variable is significant as value is less than 0.05
#R^2 value is very good as it is more than 0.80 so we can say that model is bestfit as of now
#We can write eq. as DT=7.47+0.31*ST+I(0.23*ST*ST)

#check fitted values(predicted)
reg2degree$fitted.values
reg2degree$residuals

#but we have to check with predicted values
predict(reg2degree)

#Check for error associated with each obs.
reg2degree$residuals
sum(reg2degree$residuals)

#check for mean of sum of errors is equal to 0.
mean(reg2degree$residuals)
hist(reg2degree$residuals) # check errors are normally distributed or not.

#Check for RMSE value
sqrt(sum((reg2degree$residuals)^2)/nrow(dt_st))  #RMSE
sqrt(mean(reg2degree$residuals^2))

#interval for 5% of confidence
confint(reg2degree,level=0.95)
predict(reg2degree,interval="confidence")

# visualization

ggplot(data = dt_st, aes(x = `Sorting.Time` + I((`Sorting.Time`)^2), y = `Delivery.Time`)) + 
  geom_point(color='blue') +
  geom_line(color='red',data = dt_st, aes(x=`Sorting.Time`+I(`Sorting.Time`^2), y=pred))

#Inferences-
#From all above  value or correlation coe.(r) is 0.82 & 0.79 resp as polynomial which is moderate and decreased than previuos ,
# function is linear in nature i.e. lm((DT) ~ ST+I(ST*ST), 
# Coe. are significant and coe.of Determination value (R^2) is 0.97 which is very good and improved than previous.
# mean of errors is -5.266125e-18 which is almost 0 ans errors are normally distributed.
# RMSE value is  0.7902562 increased slightly than previous model which is nearest to lower range value of delivery time
# so as model best fits,we need to go with this transformation although correlation coe.(r) value is slightly less than best fit.
# Conclusion : There were 3 influencing data by removing which the model becomes strong.
##################### problem 3#######################
library(readr)
# Build a prediction model for Churn_out_rate 

sh.cr <- read.csv(file.choose()) # choose the Emp_Data.csv data set
View(sh.cr)
# 10 Observations of 2 variables
# Scatter Diagram (Plot x,y)
plot(sh.cr$Salary_hike,sh.cr$Churn_out_rate)

# Other Exploratory data analysis and Plots
boxplot(sh.cr)

hist(sh.cr$Salary_hike)

hist(sh.cr$Churn_out_rate)

summary(sh.cr)

# Correlation coefficient value for Salary Hike and Churn_out_Date
cr<- sh.cr$Churn_out_rate
sh <- sh.cr$Salary_hike
cor(cr,sh)

# If |r| is greater than  0.85 then Co-relation is Strong(Correlation Co-efficient = -0.9117216). 
# This has a strong negative Correlation 

# Simple model without using any transformation
reg<-lm(cr~sh)
summary(reg)

# Probability value should be less than 0.05(1.96e-05)
# The multiple-R-Squared Value is 0.8312 which is greater than 0.8(In General)
# Adjusted R-Squared Value is 0.8101 
# The Probability Value for F-Statistic is 0.0002386(Overall Probability Model is also less than 0.05)
confint(reg,level = 0.95) # confidence interval

# The above code will get you 2 equations 
# 1 to caliculate the lower range and other for upper range

# Function to Predict the above model 
predict(reg,interval="predict")

# predict(reg,type="prediction")
# Adjusted R-squared value for the above model is 0.8101 

# we may have to do transformation of variables for better R-squared value
# Applying transformations

# Logarthmic transformation
reg_log<-lm(cr~log(sh))  # Regression using logarthmic transformation
summary(reg_log)

confint(reg_log,level=0.95)

predict(reg_log,interval="predict")

# Multiple R-squared value for the above model is 0.8486
# Adjusted R-squared:  0.8297 

# we may have to do different transformation for a better R-squared value
# Applying different transformations

# Exponential model 
reg_exp<-lm(log(cr)~sh) # regression using Exponential model
summary(reg_exp)

confint(reg_exp,level=0.95)

exp(predict(reg_exp,interval="predict"))

# Multiple R-squared value - 0.8735
# Adjusted R SQuare Value - 0.8577 
# Higher the R-sqaured value - Better chances of getting good model 
# for Delivery Time and Sorting Time


# Quadratic model
qd_model <- lm(cr~sh+I(sh^2),data=sh.cr)
summary(qd_model)

confint(qd_model,level=0.95)

predict(qd_model,interval="predict")

# Adjusted R-Squared = 0.9662 
#Multiple R -Squared Value = 0.9737

# Cubic model
poly_mod <- lm(cr~sh+I(sh^2)+I(sh^3),data=sh.cr)
summary(poly_mod) # 0.9893

confint(poly_mod,level=0.95)

predict(poly_mod,interval="predict")

# Adjusted R-Squared = 0.984
#Multiple R -Squared Value = 0.9893

model_R_Squared_values <- list(model=NULL,R_squared=NULL)
model_R_Squared_values[["model"]] <- c("reg","reg_log","reg_exp","qd_modle","poly_mod")
model_R_Squared_values[["R_squared"]] <- c(0.8101,0.8297,0.8577,0.9662,0.984)
Final <- cbind(model_R_Squared_values[["model"]],model_R_Squared_values[["R_squared"]])
View(model_R_Squared_values)
View(Final)

# Cubic  model gives the best Adjusted R-Squared value
predicted_Value <- predict(poly_mod)
predicted_Value

Final <- cbind(Salary_Hike=sh.cr$Salary_hike,Churn_Rate = sh.cr$Churn_out_rate,Pred_Chr_rate=predicted_Value)

View(Final)

rmse <-sqrt(mean((predicted_Value-cr)^2))
rmse
## [1] 1.0052
plot(poly_mod)
##########################problem 4######################
library(readr)
Salary_Data <- read.csv(file.choose())
attach(Salary_Data)
View(Salary_Data)
summary(Salary_Data)
cor(Salary_Data) #cor=0.9782416

library(psych)

#Linear Regression
plot(Salary_Data)
reg <- lm(Salary_Data$Salary~Salary_Data$YearsExperience)
summary(reg) #R^2=0.957
confint(reg, level = 0.95)
pred <- predict(reg, interval = "predict")
sqrt(mean(reg$residuals)^2) #RMSE=2.615537e-13
cor(pred, Salary_Data$Salary) #Accuracy=0.9782416

#Logarithmic Transformation
plot(log(YearsExperience), Salary)
reg1 <- lm(Salary_Data$Salary~log(YearsExperience))
summary(reg1) #R^2=0.8539
confint(reg1, level = 0.95)
pred1 <- predict(reg1, interval = "predict")
sqrt(mean(reg1$residuals)^2) #RMSE=1.622051e-12
cor(pred1, Salary_Data$Salary) #Accuracy=0.9240611

#Exponential Transformation
plot(log(Salary), YearsExperience)
reg2 <- lm(log(Salary)~YearsExperience)
summary(reg2) #R^2=0.932
confint(reg2, level = 0.95)
log_pred2 <- predict(reg2, interval = "predict")
pred2 <- exp(log_pred2)
sqrt(mean(reg2$residuals)^2) #RMSE=5.26335e-18
cor(pred2, Salary_Data$Salary) #Accuracy=0.9660470

#Quadratic Transformation raised to the power 2
plot(YearsExperience^2, Salary)
reg3 <- lm(Salary_Data$Salary~Salary_Data$YearsExperience + I(YearsExperience^2))
summary(reg3) #R^2=0.957
confint(reg3, level = 0.95)
pred3 <- predict(reg3, interval = "predict")
sqrt(mean(reg3$residuals)^2) #RMSE=3.202623e-13
cor(pred3, Salary_Data$Salary) #Accuracy=0.9782511

#Quadratic Transformation raised to the powe 3
plot(YearsExperience^3, Salary)
reg4 <- lm(Salary_Data$Salary~Salary_Data$YearsExperience + I(YearsExperience^2) + I(YearsExperience^3))
summary(reg4) #R^2=0.9636
confint(reg4, level = 0.95)
pred4 <- predict(reg4, interval = "predict")
sqrt(mean(reg4$residuals)^2) #RMSE=7.202757e-14
cor(pred4, Salary_Data$Salary) #Accuracy=0.9816298

#quadratic transformation raised to the power 3 posses the high R^2 value so it is the best model
plot(reg4)

# Correlation coefficient between fitted value and actual is 0.98
# This model can be considered as best model
#################### problem 5 #############################
library(readr)
sat <- read.csv(file.choose()) # choose the Sat data set
View(sat)

summary(sat)
# 30 Observations of 2 variables
attach(sat)
# Other Exploratory data analysis and Plots

boxplot(sat)

hist(sat$SAT_Scores)

hist(sat$GPA)

# Correlation coefficient value for Years of Experience and Employee Salary Hike
ye<-sat$SAT_Scores
sh <- sat$GPA
cor(sat)

# Simple model without using any transformation
reg<-lm(sh ~ye )
summary(reg)

# Probability value should be less than 0.05(5.51e-12)
# The multiple-R-Squared Value is 0.957 which is greater than 0.8(In General)
# Adjusted R-Squared Value is 0.9554 
# The Probability Value for F-Statistic is 2.2e-16(Overall Probability Model is also less than 0.05)
confint(reg,level = 0.95) # confidence interval

# The above code will get you 2 equations 
# 1 to caliculate the lower range and other for upper range
# Function to Predict the above model 
predict(reg,interval="predict")

# predict(reg,type="prediction")
# Adjusted R-squared value for the above model is 0.9554 

# we may have to do transformation of variables for better R-squared value
# Applying transformations

# Logarthmic transformation
reg_log<-lm(sh~log(ye))  # Regression using logarthmic transformation
summary(reg_log)

confint(reg_log,level=0.95)

predict(reg_log,interval="predict")

# Multiple R-squared value for the above model is 0.8539
# Adjusted R-squared:  0.8487 

# we may have to do different transformation for a better R-squared value
# Applying different transformations

# Exponential model 
reg_exp<-lm(log(sh)~ye) # regression using Exponential model
summary(reg_exp)

confint(reg_exp,level=0.95)

exp(predict(reg_exp,interval="predict"))

# Multiple R-squared value - 0.932
# Adjusted R SQuare Value - 0.9295 
# Higher the R-sqaured value - Better chances of getting good model 
# for Salary hike and Years of Experience

# Quadratic mode
sat[,"ye_sq"] = ye*ye

# Quadratic model
quad_mod <- lm(sh~ye+I(ye^2),data=sat)
summary(quad_mod)

confint(quad_mod,level=0.95)

predict(quad_mod,interval="predict")


# Quadratic model
qd_model <- lm(sh~ye+ye_sq,data=sat)
summary(qd_model)

confint(quad_mod,level=0.95)

predict(quad_mod,interval="predict")

# Adjusted R-Squared = 0.9538 
#Multiple R -Squared Value = 0.957

# Cubic model
poly_mod <- lm(sh~ye+I(ye^2)+I(ye^3),data=sat)
summary(poly_mod) # 0.9636

confint(poly_mod,level=0.95)

predict(poly_mod,interval="predict")

# Adjusted R-Squared = 0.9594
#Multiple R -Squared Value = 0.9636

model_R_Squared_values <- list(model=NULL,R_squared=NULL)
model_R_Squared_values[["model"]] <- c("reg","reg_log","reg_exp","quad_mod","poly_mod")
model_R_Squared_values[["R_squared"]] <- c(0.9554,0.8487,0.9295,0.9538,0.9594)
Final <- cbind(model_R_Squared_values[["model"]],model_R_Squared_values[["R_squared"]])
View(model_R_Squared_values)
View(Final)
