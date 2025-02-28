# Normalization

# Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.
# Suppose I have a dataset of the person weight and the weight is in kg, pound, gram. It has different units, which is not good for machine learning. We should eliminate the units. So, we need to change the data into common scale and than use machine learning. This is called normalization.

# Types of normalization
# 1. MinMaxScaling (most popular)
# 2. Mean Normalization
# 3. Max Absolute Scaling
# 4. Robust Scaling

# 1. MinMaxScaling
# Suppose I have these values, weight = [130, 67, 81, 61, 32, 54]
# From this dataset the Xmin v= 32, Xmax = 130
# MinMaxScaling formula = (Xcurrent - Xmin)/(Xmax - Xmin)
# That is for 130, (130 - 32)/(130 - 32)  = 1
# For 67, (67 - 32) / (130 - 32) = 0.3571428571
# The MinMaxScaling ranges from [0, 1].   
# The geometry meaning is that it squezes the data between a square box or 1:1

import numpy as np     # Linear Algebra
import pandas as pd    # Data Processing
import matplotlib.pyplot as plt 
import seaborn as sns 

df = pd.read_csv('wine_data.csv', header = None, usecols = [0, 1, 2])
df.columns = ['Class label', 'Alcohol', 'Malic acid']

# Start by split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Class label', axis = 1), df['Class label'], text_size = 0.3, random_state = 0)

# import MinMaxScaler class
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# fit the scaler to the train set, it will learn the parameters
# The fit should always be done on training data
scaler.fit(X_train)

# transform train and test sets
# We need to transform the both training and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# sklearn converts datafram into numpy array. Here I am converting that numpy array into the dataframe
X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)

# NOTE: Normalization will always retain the original shape. It only squeezes the data into a smaller square. The data representation is mostly always same.

# 2. Mean Normalization
# Mean Normalization formula = (Xcurrent - Xmean)/(Xmax - Xmin)
# Range is [-1, 1]
# Useful when we need centered data. Or we can also use Standaridation.

# 3. Max Absolute Scaling
# Max Absolute Scaling formula = Xcurrent / Abs(Xmax)
# There is a class in sklearn called MaxAbsScaler to perform this.
# It is useful in the sparse data. It is the data where we have lots of 0s.

# 4. Robust Scaling
# Robust Scalingformula = (Xcurrent - Xmediam) / IQR
# IQR = Inter Quartile Range {75th percentile value - 25th percentile value }
# In sklearn we have a class called Robust Scaler class. 
# If your data has lots of outliers we can use Robust Scaling

## Normalization vs Standardization ##
# 1. Is feature scaling required?







