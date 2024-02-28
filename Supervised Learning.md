## Supervised Learning Overview

**Definition:**  
Supervised learning is a core concept in machine learning where algorithms learn from labeled data, comprising input-output pairs, to predict outputs for unseen inputs.

**Key Components:**
- **Input Data:** Variables used for predictions, such as numerical or categorical features.
- **Output Labels:** Target variables for prediction, discrete (classification) or continuous (regression).
- **Training Data:** Labeled dataset used to train models, forming the basis for learning patterns.
- **Model:** Algorithm learning from training data to predict outcomes on new data.
- **Loss Function:** Measures difference between model predictions and actual labels, guiding optimization.
- **Optimization Algorithm:** Adjusts model parameters to minimize loss, often using gradient descent.
- **Evaluation Metrics:** Assess model performance on unseen data using accuracy, precision, etc.

**Types of Supervised Learning:**
- **Classification:** Predicts discrete categories like spam detection or sentiment analysis.
- **Regression:** Predicts continuous values such as house prices or temperature forecasting.

**Workflow:**
1. **Data Collection:** Gather labeled input-output pairs.
2. **Data Preprocessing:** Clean, handle missing values, and engineer features.
3. **Model Selection:** Choose an appropriate algorithm based on data and problem.
4. **Training:** Train the model to minimize loss on training data.
5. **Evaluation:** Assess model performance using validation or test datasets.
6. **Hyperparameter Tuning:** Optimize model performance further.
7. **Deployment:** Deploy model for real-world predictions.

**Applications:**
Supervised learning finds applications in:
- Email spam detection
- Handwritten digit recognition
- Medical diagnosis
- Autonomous driving
- Financial forecasting
- Natural language processing
- Recommender systems

In summary, supervised learning enables algorithms to predict outputs based on labeled data, driving advancements across diverse domains in artificial intelligence.
## Linar regression 

**Definition:**  
Supervised learning is a core concept in machine learning where algorithms learn from labeled data, comprising input-output pairs, to predict outputs for unseen inputs.

**Key Components:**
- **Input Data:** Variables used for predictions, such as numerical or categorical features.
- **Output Labels:** Target variables for prediction, discrete (classification) or continuous (regression).
- **Training Data:** Labeled dataset used to train models, forming the basis for learning patterns.
- **Model:** Algorithm learning from training data to predict outcomes on new data.
- **Loss Function:** Measures difference between model predictions and actual labels, guiding optimization.
- **Optimization Algorithm:** Adjusts model parameters to minimize loss, often using gradient descent.
- **Evaluation Metrics:** Assess model performance on unseen data using accuracy, precision, etc.

**Types of Supervised Learning:**
- **Classification:** Predicts discrete categories like spam detection or sentiment analysis.
- **Regression:** Predicts continuous values such as house prices or temperature forecasting.

**Workflow:**
1. **Data Collection:** Gather labeled input-output pairs.
2. **Data Preprocessing:** Clean, handle missing values, and engineer features.
3. **Model Selection:** Choose an appropriate algorithm based on data and problem.
4. **Training:** Train the model to minimize loss on training data.
5. **Evaluation:** Assess model performance using validation or test datasets.
6. **Hyperparameter Tuning:** Optimize model performance further.
7. **Deployment:** Deploy model for real-world predictions.

**Applications:**
Supervised learning finds applications in:
- Email spam detection
- Handwritten digit recognition
- Medical diagnosis
- Autonomous driving
- Financial forecasting
- Natural language processing
- Recommender systems

In summary, supervised learning enables algorithms to predict outputs based on labeled data, driving advancements across diverse domains in artificial intelligence.
sion

### Example problem:
***problem statement**
1.Analyse the California housing problem using linear regression method

step-1:importinhg necessary libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
```
step-2: Load the California Housing Dataset
```python
california_housing = fetch_california_housing()

# Display information about the dataset
``python
print(california_housing.DESCR)
```
Output:
.. _california_housing_dataset:

California Housing dataset
--------------------------

**Data Set Characteristics:**

    :Number of Instances: 20640

    :Number of Attributes: 8 numeric, predictive attributes and the target

    :Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude

    :Missing Attribute Values: None

This dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the
:func:`sklearn.datasets.fetch_california_housing` function.

.. topic:: References

    - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
      Statistics and Probability Letters, 33 (1997) 291-297
Step 3: Prepare the data
```pyhon
X = california_housing.data
y = california_housing.target
```
Step 4: Split the data into training and testing sets
```pyhon
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Step 5: Create and train the linear regression model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```
Step 6: Make predictions on the test set
```python
y_pred_linear = model.predict(X_test)
```
 Step 7: Evaluate the model
 ```python
mse = mean_squared_error(y_test, y_pred_linear)
print(f'Mean Squared Error: {mse}')
```
Output:
Mean Squared Error: 0.555891598695197
Step 8: Visualize the results
```python
plt.scatter(y_test, y_pred_linear)
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Actual vs. Predicted House Prices')
plt.show()
```
Output:![a1](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/14617078-d92c-4bbf-9eb0-f86260a25b75)

```python
# R2
r2 = r2_score(y_test, y_pred_linear)
print("R2 --> ", r2)
```
Output:
      R2 -->  0.575787706032487

 Step 8: Apply L1 (Lasso) regularization
 ```python
lasso_model = Lasso(alpha=0.01)  # You can adjust the alpha parameter
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
```
Step 9: Evaluate the Ridge model
```python
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f'Ridge Regression - Mean Squared Error: {mse_ridge}')
```
Output:Ridge Regression - Mean Squared Error: 0.55589071394375
Step 10: Apply Elastic Net regularization
```python
elastic_net_model = ElasticNet(alpha=0.01, l1_ratio=0.6)  # You can adjust the alpha and l1_ratio parameters
elastic_net_model.fit(X_train, y_train)
y_pred_elastic_net = elastic_net_model.predict(X_test)
```
Step 10: Evaluate the Elastic Net model
```python
mse_elastic_net = mean_squared_error(y_test, y_pred_elastic_net)
print(f'Elastic Net Regression - Mean Squared Error: {mse_elastic_net}')
```
Output:Elastic Net Regression - Mean Squared Error: 0.54536192790266
```pyhton
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred_linear)
plt.title('Linear Regression - Actual vs. Predicted')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred_lasso)
plt.title('Lasso Regression - Actual vs. Predicted')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred_ridge)
plt.title('Ridge Regression - Actual vs. Predicted')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_elastic_net)
plt.title('Elastic Net Regression - Actual vs. Predicted')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')

plt.tight_layout()
plt.show()
```
Output:![a2](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/f744d737-6326-4315-bece-c701f09b17e7)
![a2](https://github.com/Nandana-Ajayan/Python-for-ML/assets/160465008/f6bcf23f-483c-4da3-b8ba-94efb989d040)

### Example problem:
***problem statement**
2.Analyse the car data  problem using linear regression method
#importing pickle to save the model

```python
import pickle as pickle
```
# Specify the path to your CSV file
```python
csv_file_path = 'car_dataset.csv'
```
# import all necessary packages
```python
import pandas as pd
```
# load data to a new dataframe
```python
df = pd.read_csv(csv_file_path)
```
# check the shape
```pyhton
df.shape
```
Output:
(205, 26)

# first five rows of the dataframe
```python
df.head()
```
Output:

# describe the dataframe with some statistical info
```python
df.describe()
```
Output:
# check data types in the dataframe
```pyhton
df.info()
```
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 205 entries, 0 to 204
Data columns (total 26 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   ID                205 non-null    int64  
 1   symboling         205 non-null    int64  
 2   name              205 non-null    object 
 3   fueltypes         205 non-null    object 
 4   aspiration        205 non-null    object 
 5   doornumbers       205 non-null    object 
 6   carbody           205 non-null    object 
 7   drivewheels       205 non-null    object 
 8   enginelocation    205 non-null    object 
 9   wheelbase         205 non-null    float64
 10  carlength         205 non-null    float64
 11  carwidth          205 non-null    float64
 12  carheight         205 non-null    float64
 13  curbweight        205 non-null    int64  
 14  enginetype        205 non-null    object 
 15  cylindernumber    205 non-null    object 
 16  enginesize        205 non-null    int64  
 17  fuelsystem        205 non-null    object 
 18  boreratio         205 non-null    float64
 19  stroke            205 non-null    float64
 20  compressionratio  205 non-null    float64
 21  horsepower        205 non-null    int64  
 22  peakrpm           205 non-null    int64  
 23  citympg           205 non-null    int64  
 24  highwaympg        205 non-null    int64  
 25  price             205 non-null    float64
dtypes: float64(8), int64(8), object(10)
memory usage: 41.8+ KB
# check unique data for each feature in the dataframe
```python
df.nunique()
```
Output:
ID                  205
symboling             6
name                147
fueltypes             2
aspiration            2
doornumbers           2
carbody               5
drivewheels           3
enginelocation        2
wheelbase            53
carlength            75
carwidth             44
carheight            49
curbweight          171
enginetype            7
cylindernumber        7
enginesize           44
fuelsystem            8
boreratio            38
stroke               37
compressionratio     32
horsepower           59
peakrpm              23
citympg              29
highwaympg           30
price               189
dtype: int64
# column names of the dataframe
```python
df.columns()
```
Output:
Index(['ID', 'symboling', 'name', 'fueltypes', 'aspiration', 'doornumbers',
       'carbody', 'drivewheels', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio',
       'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price'],
      dtype='object')


### Now, we can prepare the data for the linear regression model
# create a new simple dataframe using the existing one
```python
new_df = df[['enginesize', 'price']]
new_df
```
Output:
# check the distribution of data by plotting as scatter
```python
new_df.plot(x='enginesize', y='price', kind='scatter')
```
Output:
# compare below plot with the previous one
```python
df.plot(x='fueltypes', y='price', kind='scatter')
```
Output:
# define x and y by using the new dataframe
```python
x = new_df['enginesize']
y = new_df['price']
```
### Now Machine Learning
# import ML related packages of sklearn
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```
# split the data as train and test
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
```
#shapes of splitted data - sizes of the X_train and Y_train should be same. Also, tests.
```python
print("X_train:",x_train.shape)
print("X_test:",x_test.shape)
print("Y_train:",y_train.shape)
print("Y_test:",y_test.shape)
```
Output:
X_train: (164,)
X_test: (41,)
Y_train: (164,)
Y_test: (41,)

# create a linear regression model
```python
model = LinearRegression()
```
# train the model using training data
 since we have only one independent variable, you should use 'values.reshape(-1,1)'. Otherwise, x_train is enough.
```python
model.fit(x_train.values.reshape(-1,1), y_train)
```
# print coefficient
```python
model.coef_
```
Output:
array([164.31545172])

# print y_intercept | bias
```python
model.intercept_
```
Output:
-7613.370939489788

# make predictions using test data
```python
y_pred = model.predict(x_test.values.reshape(-1,1))
```
# let's calculate the metrics
 MSE
 ```pyhton
mse = mean_squared_error(y_test, y_pred)
print("MSE --> ", mse)
```
Output:
MSE -->  16835544.03813768

# RMSE
```python
import math
rmse = math.sqrt(mse)
print("RMSE --> ", rmse)
```
Output:
RMSE -->  4103.113944084137

# MAE
```python
mae = mean_absolute_erroRMSE -->  4103.113944084137r(y_test, y_pred)
print("MAE --> ", mae)
```
MAE -->  3195.0312395000437
