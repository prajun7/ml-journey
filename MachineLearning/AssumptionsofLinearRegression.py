# Assumptions of Linear Regression

# YT Video: https://www.youtube.com/watch?v=EmSNAtcHLm8&list=PPSV&t=2s
# YT Title: What are the main Assumptions of Linear Regression? | Top 5 Assumptions of Linear Regression

# 1. Linear Relationship between input and output
# 2. No Multicollinearity
# 3. Normality of Residual
# 4. Homoscedasticity
# 5. No Autocorrelation of Errors

###### 1. Linear Relationship between input and output
# The data must have a linear relationship to perform Linear Regression. That is as the x axis increases or decreases the y axis shouuld also increase or decrease accordingly. If y axis is increasing in twice the amount as x axsis it is a quadractice realtiona dn not a linear relationship. And hence we cannot perform linear regression.
# We can plot a graph in jupyter to figure this out

###### 2. No Multicollinearity
# If we have a data like x1, x2, x3 as the input column and y as the output(target) column. Than x1, x2, x3 should be independednt and they should not depend on each other. They should have no corelation. Id we increase x1 than if x2 or x3 also increases than we call that relation to be multicollinearity. And for linear regression there should not be multicollinearity.
# We can use variance_inflation_factor to determine if our data has multicollinearity or not. If we are getting 1 than we don't have multicollinearity but if we get 5 or above 5 than we have multicollinearity.

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = []

for i in range(x_train.shape[1]):
	vif.append(variance_inflation_factor(x_train, i))

pd.DataFrame({'vif': vif}, index = df.columns[0:3]), T

# Another Technique
sns.heatmap(df.iloc[:, 0:3].corr(), annot=True)
# This will draw a heatmap graoh and we can check the correlation between any two feature. If the correlation value is low that means they don't have any relationship.

###### 3. Normality of Residual
# Residual are error. So when we plot the residual the graph should be normal that is the mean should be close to 0. We can plot KDE or QQ to figure this out
y_pred = model.predict(x_test)
residual = y_test - y_pred
# calcualates the residual or the errors

sns.displot(residual, kind = 'kde')
# plots the graph

# OR plit QQ Plt
import scipy as sp
fig, ax = plt.subplots(figsize=(6,4))
sp.stats.probplot(residual, plot = ax, fit = True)
plt.show()


###### 4. Homoscedasticity (Having the same scatter)
# When we plot the residual, its spread should be equal. If it is not equal it is called Hetroscedasticity and this is not allowed in Linear Regression. We can Homoscedasticity.


###### 5. No Autocorrelation of Errors
# When we plot the residual value it should not create any pattern. 







