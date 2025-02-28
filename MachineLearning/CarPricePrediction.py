# Machine Learning: Car Price Predictor | Linear Regressiion

# YT Link: https://www.youtube.com/watch?v=iRCaMnR_bpA&list=PPSV&t=4248s
# YT Title: Car Price Predictor Project | Machine Learning | Linear Regression
# YT Chennel: CampusX

# Read the csv file that has the car information
import pandas as pd 
car = pd.read_csv('quikr_car.csv')

# Display those car data
car.head()

# Displays how many rows and columns we have
# OUtput: (892, 6), that is 892 rows and 6 columns
car.shape

# To see the info of the car data. It displays the column type
car.info()

# Need to clean the data
# year column is object but it should be int. Also, some years values are text. Need to clean those.
# price has some string. Need to remove this. Also, Price is object, need to convert into int
# kms_driven has kms string. Also has null values, 
# fuel_type has null values
# Keep first 3 words of the car


# Only shows the unique 
car['Year'].unique()


# CLeaning
# Backup the original car data
backup = car.copy()

# Remove those years that has string value
car['year'].str.isNumeric()
# This returns true if the car year is numeric else false

car[car['year'].str.isNumeric()]
# Here only those cars will be shown that has numeric values

car = car[car['year'].str.isNumeric()]
# Srore the updated value back into the car

car['year'] = car['year'].astype(int)
# Changing the str to int value for all the years

# Remove those price that has string values. The string value that is present in the year column is "Ask For Price". So, we will reemove this year that have Ask for Price
car = car[car['price'] != "Ask For Price"]

# Replace commas by empty string in price column
car['Price'] = car['Price'].str.replace(',', '').astype(int)

# Clean kms_driven by remving commas and kms value
# Input: 80,000 kms
# Output: 80000
car['kms_driven'] = car['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')

# Also some kms_driven has a string, so we will remove that as well
car = car[car['kms_driven'].str.isNumeric()]

# Chaning kms_driven into int
car['kms_driven'] = car['kms_driven'].astype(int)

# Removing the NaN value from the fuel_type
car = car[~car['fuel_type'].isna()]

# Only getting the first three words of the name of the car
car['name'] = car['name'].str.split(' ').str.slice(0, 3).join(' ')

# Since we deleted some of the car values, so some of the indexes are missing. We can use reset_index to fix that
car = car.reset_index(drop = True)

# Store the clean car value into the csv
car.to_csv('CleanedCar.csv')


# Model
# Extract the feature(x) and labels(y).
# Everything is our feature except the price column
x = car.drop(cplumns = 'Price')
y = car['Price']

# Apply renders split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

ohe = OneHotEncoder()
ohe.fit(x[['name', 'company', 'fule_type']])

column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_), ['name', 'company', 'fuel_type']), remainder='passthrough')

lr = LinearRegression()

pipe = make_pipeline(column_trans, lr)

y_pred = pipe.predict(x_test)
 

























