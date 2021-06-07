######################## problem 1#######################
import pandas as pd
#loading the dataset
cal = pd.read_csv("C:/Users/usach/Desktop/Simple Linear Regression/calories_consumed.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary
description  = ["important data weight gained in grams ",
                "Consumptation in Calories"]

d_types =["Ratio","Ratio"]

data_details =pd.DataFrame({"column name":cal.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": cal.dtypes})

#3.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc
#details of cal 
cal.info()
cal.describe()          
#rename the columns
cal.rename(columns = {'Weight gained (grams)':'weight', 'Calories Consumed':'calories'}, inplace = True) 
#data types        
cal.dtypes
#checking for na value
cal.isna().sum()
cal.isnull().sum()
#checking unique value for each columns
cal.nunique()

"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    
EDA ={"column ": cal.columns,
      "mean": cal.mean(),
      "median":cal.median(),
      "mode":cal.mode(),
      "standard deviation": cal.std(),
      "variance":cal.var(),
      "skewness":cal.skew(),
      "kurtosis":cal.kurt()}

EDA

# covariance for data set 
covariance = cal.cov()
covariance
####### graphical repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(cal.iloc[:, :])
#boxplot for every columns
cal.columns
cal.nunique()
cal.boxplot(column=['weight', 'calories'])   #no outlier
"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and calculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """
import numpy as np
#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(cal.calories, cal.weight)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf

model1= smf.ols('calories ~ weight' , data = cal).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(cal['weight']))
pred
import matplotlib.pyplot as plt

# Regression Line
plt.scatter(cal.weight, cal.calories)
plt.plot(cal.weight, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

import numpy as np
# Error calculation
rmse =  np.sqrt(((pred-cal['calories'])**2).mean())
rmse

#model 2
# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation
plt.scatter(x = np.log(cal['calories']), y = cal["weight"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(cal.calories), cal.weight) #correlation
Co_coe_val_2

model2 = smf.ols('calories ~ np.log(weight)', data =cal).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(cal["weight"]))

# Regression Line

plt.scatter(np.log(cal.weight),cal.calories)
plt.plot(np.log(cal.weight), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
rmse2 =  np.sqrt(((pred2-cal['calories'])**2).mean()) 
rmse2

# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = cal['calories'], y = np.log(cal['weight']), color = 'orange') 
Co_coe_val_2  =    np.corrcoef(cal.calories, np.log(cal['weight'])) #correlation
Co_coe_val_2  
#model3

model3 = smf.ols('np.log(cal.calories) ~ cal.weight ', data = cal).fit() 
model3.summary() 

pred3 = model3.predict(pd.DataFrame(cal['weight']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line

plt.scatter(cal['weight'], np.log(cal['calories'])) 
plt.plot(cal['weight'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error calculation
rmse3 =  np.sqrt(((pred3_at-cal['calories'])**2).mean())
rmse3   

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(calories) ~ weight + I(weight*weight)', data = cal).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(cal.weight)) 
pred4_at = np.exp(pred4) 
pred4_at  
# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = cal.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(cal.weight, np.log(cal.calories)) 
plt.plot(cal['weight'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 
# Error calculation
rmse4 =  np.sqrt(((pred4_at -cal['calories'])**2).mean()) 
rmse4
# Choose the best model using RMSE
Model_details = pd.DataFrame({"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4]),
        "R-squared": pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model4.rsquared]),
        "Adj. R-squared" : pd.Series([model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,model4.rsquared_adj])})

Model_details
###################
# The best model
from sklearn.model_selection import train_test_split

train , test = train_test_split(cal, test_size = 0.5 , random_state = 7)

finalmodel = smf.ols('calories ~ weight', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
test_pred 
# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.calories)**2).mean())
test_rmse
# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.calories)**2).mean())
train_rmse

# Result
## Applying transformation is decreasing Multiple R Squared Value. 
###So model doesnot need further transformation, Multiple R-squared:  0.911
#########################problem 2##############################
import pandas as pd 
#loading the dataset
delivery = pd.read_csv("C:/Users/usach/Desktop/Simple Linear Regression/delivery_time.csv") 

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary

description  = ["delivery time taken, important data",
                "the time sorted by the restaurants, important data"]

d_types =["Ratio","Count"]

data_details =pd.DataFrame({"column name":delivery.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": delivery.dtypes})

#3.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc
#details of delivery 
delivery.info()
delivery.describe()          
#rename the columns
delivery.rename(columns = {'Delivery Time':'Delivery_Time', 'Sorting Time':'Sorting_Time'}, inplace = True) 
#data types        
delivery.dtypes
#checking for na value
delivery.isna().sum()
delivery.isnull().sum()
#checking unique value for each columns
delivery.nunique()

"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    
EDA ={"column ": delivery.columns,
      "mean": delivery.mean(),
      "median":delivery.median(),
      "mode":delivery.mode(),
      "standard deviation": delivery.std(),
      "variance":delivery.var(),
      "skewness":delivery.skew(),
      "kurtosis":delivery.kurt()}

EDA
# covariance for data set 
covariance = delivery.cov()
covariance
####### graphidelivery repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(delivery.iloc[:, :])
#boxplot for every columns
delivery.columns
delivery.nunique()
delivery.boxplot(column=["Delivery Time"])   #no outlier
"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and deliveryculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """

import numpy as np

#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(delivery.Delivery_Time, delivery.Sorting_Time)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf
model1= smf.ols('Delivery_Time ~ Sorting_Time' , data = delivery).fit()
model1.summary()
#perdicting on whole data
pred = model1.predict(pd.DataFrame(delivery['Sorting_Time']))

import matplotlib.pyplot as plt

# Regression Line
plt.scatter(delivery.Sorting_Time, delivery.Delivery_Time)
plt.plot(delivery.Sorting_Time, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error deliveryculation
rmse =  np.sqrt(((pred-delivery['Delivery_Time'])**2).mean())
rmse
#model 2
# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation
plt.scatter(x = np.log(delivery['Delivery_Time']), y = delivery["Sorting_Time"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(delivery.Delivery_Time), delivery.Sorting_Time) #correlation
Co_coe_val_2

model2 = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data =delivery).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(delivery["Sorting_Time"]))

# Regression Line

plt.scatter(np.log(delivery.Sorting_Time),delivery.Delivery_Time)
plt.plot(np.log(delivery.Sorting_Time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error deliveryculation
rmse2 =  np.sqrt(((pred2-delivery['Delivery_Time'])**2).mean()) 
rmse2
#model3
# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 
plt.scatter(x = delivery['Delivery_Time'], y = np.log(delivery['Sorting_Time']), color = 'orange') 
Co_coe_val_3  =    np.corrcoef(delivery.Delivery_Time, np.log(delivery['Sorting_Time'])) #correlation
Co_coe_val_3  

model3 = smf.ols('np.log(delivery.Delivery_Time) ~ delivery.Sorting_Time ', data = delivery).fit() 
model3.summary() 

pred3 = model3.predict(pd.DataFrame(delivery['Sorting_Time']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line
plt.scatter(delivery['Sorting_Time'], np.log(delivery['Delivery_Time'])) 
plt.plot(delivery['Sorting_Time'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error deliveryculation
rmse3 =  np.sqrt(((pred3_at-delivery['Delivery_Time'])**2).mean())
rmse3   

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
plt.scatter(x = np.log(delivery.Delivery_Time), y = delivery.Sorting_Time, color = 'orange') 
Co_coe_val_4  =    np.corrcoef( np.log(delivery['Delivery_Time']), delivery.Sorting_Time) #correlation
Co_coe_val_4
#model 4
model4 = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time)', data = delivery).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(delivery.Sorting_Time)) 
pred4_at = np.exp(pred4) 
pred4_at  
# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = delivery.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(delivery.Sorting_Time, np.log(delivery.Delivery_Time)) 
plt.plot(delivery['Sorting_Time'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 
# Error deliveryculation
rmse4 =  np.sqrt(((pred4_at -delivery['Delivery_Time'])**2).mean()) 
rmse
# Choose the best model using RMSE
Model_details = pd.DataFrame({"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4]),
        "R-squared": pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model4.rsquared]),
        "Adj. R-squared" : pd.Series([model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,model4.rsquared_adj]),
         "Correlation coefficient values ":pd.Series([Co_coe_val_1,Co_coe_val_2,Co_coe_val_3,Co_coe_val_4])})
         
Model_details
###################
# The best model
from sklearn.model_selection import train_test_split

train , test = train_test_split(delivery, test_size = 0.5 , random_state = 775)

#final model
finalmodel = smf.ols('np.log(Delivery_Time) ~ Sorting_Time + I(Sorting_Time*Sorting_Time) ', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred_exp = finalmodel.predict(pd.DataFrame(test))
test_pred= np.exp(test_pred_exp)
test_pred
# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.Delivery_Time)**2).mean())
test_rmse
# Prediction on train data
train_pred_exp = finalmodel.predict(pd.DataFrame(train))
train_pred= np.exp(train_pred_exp)
train_pred
# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.Delivery_Time)**2).mean())
train_rmse
####################### problem 3 ########################
import pandas as pd 
#loading the dataset
churn = pd.read_csv("C:/Users/usach/Desktop/Simple Linear Regression/emp_data.csv") 
#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image

#######feature of the dataset to create a data dictionary
description  = ["data regarding the employee’s salary, important data",
                "employee churn, important data"]
d_types =["Ratio","Count"]
data_details =pd.DataFrame({"column name":churn.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": churn.dtypes})

#3.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc
#details of churn 
churn.info()
churn.describe()          
#data types        
churn.dtypes
#checking for na value
churn.isna().sum()
churn.isnull().sum()
#checking unique value for each columns
churn.nunique()
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """
    
EDA ={"column ": churn.columns,
      "mean": churn.mean(),
      "median":churn.median(),
      "mode":churn.mode(),
      "standard deviation": churn.std(),
      "variance":churn.var(),
      "skewness":churn.skew(),
      "kurtosis":churn.kurt()}

EDA
# covariance for data set 
covariance = churn.cov()
covariance
####### graphichurn repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(churn.iloc[:, :])
#boxplot for every columns
churn.columns
churn.nunique()
churn.boxplot(column=['Salary_hike', 'Churn_out_rate'])   #no outlier

"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and churnculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """

import numpy as np
#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(churn.Churn_out_rate, churn.Salary_hike)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf
model1= smf.ols('Churn_out_rate ~ Salary_hike' , data = churn).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(churn['Salary_hike']))

import matplotlib.pyplot as plt
# Regression Line
plt.scatter(churn.Salary_hike, churn.Churn_out_rate)
plt.plot(churn.Salary_hike, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error churnculation
rmse =  np.sqrt(((pred-churn['Churn_out_rate'])**2).mean())
rmse
#model 2
# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation
plt.scatter(x = np.log(churn['Churn_out_rate']), y = churn["Salary_hike"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(churn.Churn_out_rate), churn.Salary_hike) #correlation
Co_coe_val_2

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data =churn).fit()
model2.summary()
pred2 = model2.predict(pd.DataFrame(churn["Salary_hike"]))

# Regression Line
plt.scatter(np.log(churn.Salary_hike),churn.Churn_out_rate)
plt.plot(np.log(churn.Salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error churnculation
rmse2 =  np.sqrt(((pred2-churn['Churn_out_rate'])**2).mean()) 
rmse2
#model3
# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = churn['Churn_out_rate'], y = np.log(churn['Salary_hike']), color = 'orange') 
Co_coe_val_3  =    np.corrcoef(churn.Churn_out_rate, np.log(churn['Salary_hike'])) #correlation
Co_coe_val_3  

model3 = smf.ols('np.log(churn.Churn_out_rate) ~ churn.Salary_hike ', data = churn).fit() 
model3.summary() 

pred3 = model3.predict(pd.DataFrame(churn['Salary_hike']))
pred3_at = np.exp(pred3) 
pred3_at 
# Regression Line
plt.scatter(churn['Salary_hike'], np.log(churn['Churn_out_rate'])) 
plt.plot(churn['Salary_hike'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error churnculation
rmse3 =  np.sqrt(((pred3_at-churn['Churn_out_rate'])**2).mean())
rmse3   


##model4

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

plt.scatter(x = np.log(churn.Churn_out_rate), y = churn.Salary_hike, color = 'orange') 
Co_coe_val_4  =    np.corrcoef( np.log(churn['Churn_out_rate']), churn.Salary_hike) #correlation
Co_coe_val_4

model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = churn).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(churn.Salary_hike)) 
pred4_at = np.exp(pred4) 
pred4_at  
# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = churn.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values

plt.scatter(churn.Salary_hike, np.log(churn.Churn_out_rate)) 
plt.plot(churn['Salary_hike'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 

# Error churnculation
rmse4 =  np.sqrt(((pred4_at -churn['Churn_out_rate'])**2).mean()) 
rmse4

# Choose the best model using RMSE
Model_details = pd.DataFrame({"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4]),
        "R-squared": pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model4.rsquared]),
        "Adj. R-squared" : pd.Series([model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,model4.rsquared_adj]),
         "Correlation coefficient values ":pd.Series([Co_coe_val_1,Co_coe_val_2,Co_coe_val_3,Co_coe_val_4])})
         
Model_details
###################
# The best model
from sklearn.model_selection import train_test_split

train , test = train_test_split(churn, test_size = 0.5 , random_state = 775)

#final model
finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike) ', data = train).fit()
finalmodel.summary()
# Predict on test data
test_pred_exp = finalmodel.predict(pd.DataFrame(test))
test_pred= np.exp(test_pred_exp)
test_pred
# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.Churn_out_rate)**2).mean())
test_rmse
# Prediction on train data
train_pred_exp = finalmodel.predict(pd.DataFrame(train))
train_pred= np.exp(train_pred_exp)
train_pred
# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.Churn_out_rate)**2).mean())
train_rmse
#########################problem 4#############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# ############################# Importing the dataset ################################################
dataset = pd.read_csv(r'C:/Users/usach/Desktop/Simple Linear Regression/Salary_Data.csv')
print(dataset)

# ############################# Exploring data using functions #######################################
dataset.shape # 30 rows, 2 columns
dataset.columns #Years of experience, Salary
dataset.info() 
dataset.describe()
dataset.isnull().sum() # No null values
dataset.head()

# ############################# Exploring data using graphs (Data Visualization) #####################
# 1) Box plot - To know max, min, median, 25 percentile, 75 percentile
plt.boxplot(dataset["YearsExperience"])
plt.title("YearsExperience")
plt.show()
plt.boxplot(dataset["Salary"])
plt.title("Salary")
plt.show()

# 2) Histogram - To know distribution of columns
plt.hist(dataset["YearsExperience"])
plt.title("YearsExperience")
plt.show()

plt.hist(dataset["Salary"])
plt.title("Salary")
plt.show()

# 3) Scater plot - To know relation between two
plt.scatter(dataset["YearsExperience"], dataset["Salary"])
plt.title("Relation")
plt.show()
#Observation - Appears to be linear relation

# 4) Scatter Matrix - Best to know relation
scatter_matrix(dataset,figsize=(8,8))
plt.show()


# ############################# To know co-relation ##################################################
corr_matrix = dataset.corr()
print(corr_matrix)

# ############################# Splitting the data into train and test dataset ######################
X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
y = dataset.iloc[:, 1].values #get array of dataset in  1st column

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# ############################# Applying Machine Learning Algorithm ################################

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Visualizing the Training set results

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary VS Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Predicting the result of 5 Years Experience
y_pred = regressor.predict(np.array(5).reshape(-1,1))
y_pred

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

#Checking the accuracy
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#R-squared score

print('R-squared score:', regressor.score(X_test, y_test))  
######################### problem 5###########################
import pandas as pd
#loading the dataset
sat = pd.read_csv("C:/Users/usach/Desktop/Simple Linear Regression/SAT_GPA.csv")

#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary
description  = ["SAT scores based on the exam giver’s GPA ",
                "GPA, or Grade Point Average, is a number that indicates how well or how high you scored in your courses on average"]

d_types =["Ratio","Ratio"]

data_details =pd.DataFrame({"column name":sat.columns,
                            "data types ":d_types,
                            "description":description,
                            "data type(in Python)": sat.dtypes})

#3.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc
#details of sat 
sat.info()
sat.describe()          
#data types        
sat.dtypes
#checking for na value
sat.isna().sum()
sat.isnull().sum()
#checking unique value for each columns
sat.nunique()
"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """

EDA ={"column ": sat.columns,
      "mean": sat.mean(),
      "median":sat.median(),
      "mode":sat.mode(),
      "standard deviation": sat.std(),
      "variance":sat.var(),
      "skewness":sat.skew(),
      "kurtosis":sat.kurt()}

EDA
# covariance for data set 
covariance = sat.cov()
covariance

####### graphisat repersentation 
##historgam and scatter plot
import seaborn as sns
sns.pairplot(sat.iloc[:, :])

#boxplot for every columns
sat.columns
sat.nunique()
sat.boxplot(column=['SAT_Scores', 'GPA'])   #no outlier
"""
5.	Model Building:
5.1	Perform Simple Linear Regression on the given datasets
5.2	Apply different transformations such as exponential, log, polynomial transformations and satculate RMSE values, R-Squared values, Correlation Coefficient for each model
5.3	Build the models and choose the best fit model
5.4	Briefly explain the model output in the documentation	 
 """
import numpy as np
#model bulding 
# Linear Regression model
Co_coe_val_1  = np.corrcoef(sat.SAT_Scores, sat.GPA)
Co_coe_val_1
# Import library
import statsmodels.formula.api as smf
#model 1
model1= smf.ols('SAT_Scores ~ GPA' , data = sat).fit()
model1.summary()

#perdicting on whole data
pred = model1.predict(pd.DataFrame(sat['GPA']))

import matplotlib.pyplot as plt

# Regression Line
plt.scatter(sat.GPA, sat.SAT_Scores)
plt.plot(sat.GPA, pred, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()
# Error satculation
rmse =  np.sqrt(((pred-sat['SAT_Scores'])**2).mean())
rmse

#model 2
# Transformation Techniques
# Log transformation applied on 'x'
# input = log(x); output = y
######### Model building on Transformed Data
# Log Transformation

plt.scatter(x = np.log(sat['SAT_Scores']), y = sat["GPA"], color = 'brown')
Co_coe_val_2  =np.corrcoef(np.log(sat.SAT_Scores), sat.GPA) #correlation
Co_coe_val_2

model2 = smf.ols('SAT_Scores ~ np.log(GPA)', data =sat).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(sat["GPA"]))

# Regression Line
plt.scatter(np.log(sat.GPA),sat.SAT_Scores)
plt.plot(np.log(sat.GPA), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error satculation
rmse2 =  np.sqrt(((pred2-sat['SAT_Scores'])**2).mean()) 
rmse2
#model3
# Log transformation applied on 'y'
# input = x; output = log(y) 
#### Exponential transformation 
# x = waist; y = log(at) 

plt.scatter(x = sat['SAT_Scores'], y = np.log(sat['GPA']), color = 'orange') 
Co_coe_val_3  =    np.corrcoef(sat.SAT_Scores, np.log(sat['GPA'])) #correlation
Co_coe_val_3  

model3 = smf.ols('np.log(sat.SAT_Scores) ~ sat.GPA ', data = sat).fit() 
model3.summary() 

pred3 = model3.predict(pd.DataFrame(sat['GPA']))
pred3_at = np.exp(pred3) 
pred3_at 

# Regression Line

plt.scatter(sat['GPA'], np.log(sat['SAT_Scores'])) 
plt.plot(sat['GPA'], pred3, "r") 
plt.legend(['Predicted line', 'Observed data']) 
plt.show()

# Error satculation
rmse3 =  np.sqrt(((pred3_at-sat['SAT_Scores'])**2).mean())
rmse3   

##model4
#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

plt.scatter(x = np.log(sat.SAT_Scores), y = sat.GPA, color = 'orange') 
Co_coe_val_4  =    np.corrcoef( np.log(sat['SAT_Scores']), sat.GPA) #correlation
Co_coe_val_4

model4 = smf.ols('np.log(SAT_Scores) ~ GPA + I(GPA*GPA)', data = sat).fit() 
model4.summary() 

pred4 = model4.predict(pd.DataFrame(sat.GPA)) 
pred4_at = np.exp(pred4) 
pred4_at  
# Regression line
from sklearn.preprocessing import PolynomialFeatures  
poly_reg = PolynomialFeatures(degree = 2)  
X = sat.iloc[:, 0:1].values  
X_poly = poly_reg.fit_transform(X) 
# y = wcat.iloc[:, 1].values
plt.scatter(sat.GPA, np.log(sat.SAT_Scores)) 
plt.plot(sat['GPA'], pred4, color = 'red') 
plt.legend(['Predicted line', 'Observed data']) 
plt.show() 
# Error satculation
rmse4 =  np.sqrt(((pred4_at -sat['SAT_Scores'])**2).mean()) 
rmse4
# Choose the best model using RMSE
Model_details = pd.DataFrame({"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), 
        "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4]),
        "R-squared": pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model4.rsquared]),
        "Adj. R-squared" : pd.Series([model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,model4.rsquared_adj]),
         "Correlation coefficient values ":pd.Series([Co_coe_val_1,Co_coe_val_2,Co_coe_val_3,Co_coe_val_4])})
         
Model_details
###################
# The best model
from sklearn.model_selection import train_test_split

train , test = train_test_split(sat, test_size = 0.7 , random_state = 7)

finalmodel = smf.ols('SAT_Scores ~ GPA', data = train).fit()
finalmodel.summary()
# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
# Model Evaluation on Test data
test_rmse = np.sqrt(((test_pred-test.SAT_Scores)**2).mean())
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
# Model Evaluation on train data
train_rmse = np.sqrt(((train_pred-train.SAT_Scores)**2).mean())
train_rmse
