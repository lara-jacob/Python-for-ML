# ANALYSING THE FEEDBACK OF INTEL UNNATI SESSIONS USING ML TECHNIQUES


## INTRODUCTION
       
Feedback analysis is the methodical examination of feedback data, involving processes for collection, processing, and interpretation to extract actionable insights. Its relevance lies in its capacity to offer invaluable insights, foster continuous improvement, and drive informed decision-making across diverse domains.Moreover, feedback analysis often involves the implementation of feedback loops to enable continuous improvement and iterative refinement based on ongoing input. 
       
### Step-1:Importing necessary libraries
The required libraries such as `pandas,numpy,seaborn,mathplotlib` etc are imported.

NB:Install the libraries if not present,using ``` pip install ``` corresponding library name.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
### Step-2:Loading the data to be analysed
The dataset can be in the form of csv file or may be directly from another sites.Specify the path of the dataset that we need to upload.

```
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```
```
df_class.head()
```
OUTPUT:
![extracting tables and cluster centers](https://github.com/lara-jacob/Python-for-ML/assets/160465136/19590925-16ec-4d14-ab69-6374dc0fae78)

To make the table more attractive we can use the following commands:
```python
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen',
                           'color': 'white',
                           'border-color': 'darkblack'})
```
OUTPUT:
![bcolor data](https://github.com/lara-jacob/Python-for-ML/assets/160465136/9dbf69c2-f246-4a02-9b67-637a3215db97)

### Step-3:Optimizing the dataset
Processing the data before tarining by removing the unnecessary columns.
```python
df_class.info()
```
In pandas, the `info()` method is used to get a concise summary of a DataFrame. This method provides information about the DataFrame, including the data types of each column, the number of non-null values, and memory usage. It's a handy tool for quickly assessing the structure and content of your DataFrame.

#### Simple Breakdown of `info()` Method:

- **Index and Datatype of Each Column:** Shows the name of each column along with the data type of its elements (e.g., int64, float64, object).

- **Non-Null Count:** Indicates the number of non-null (non-missing) values in each column.

- **Memory Usage:** Provides an estimate of the memory usage of the DataFrame.

This method is especially useful when you want to check for missing values, understand the data types in your DataFrame, and get an overall sense of its size and composition.
It's often used as a first step in exploring and understanding the characteristics of a dataset.
OUTPUT:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 174 entries, 0 to 173
Data columns (total 12 columns):
 #   Column                                                                                                                                                                                 Non-Null Count  Dtype 
---  ------                                                                                                                                                                                 --------------  ----- 
 0   Timestamp                                                                                                                                                                              174 non-null    object
 1   Name of the Participant                                                                                                                                                                174 non-null    object
 2   Email ID                                                                                                                                                                               174 non-null    object
 3   Branch                                                                                                                                                                                 174 non-null    object
 4   Semester                                                                                                                                                                               174 non-null    object
 5   Recourse Person of the session                                                                                                                                                         174 non-null    object
 6   How would you rate the overall quality and relevance of the course content presented in this session?                                                                                  174 non-null    int64 
 7   To what extent did you find the training methods and delivery style effective in helping you understand the concepts presented?                                                        174 non-null    int64 
 8   How would you rate the resource person's knowledge and expertise in the subject matter covered during this session?                                                                    174 non-null    int64 
 9   To what extent do you believe the content covered in this session is relevant and applicable to real-world industry scenarios?                                                         174 non-null    int64 
 10  How would you rate the overall organization of the session, including time management, clarity of instructions, and interactive elements?                                              174 non-null    int64 
 11  Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.  10 non-null     object
dtypes: int64(5), object(7)
memory usage: 16.4+ KB

#### removing unnecessary columns
```python
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
```
#### specifying column names
```python
df_class.columns = ["Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
df_class.sample(5)
```
output:
#### checking for null values and knowing the dimensions
```python
df_class.isnull().sum().sum()
```
Output:0
```python
# dimension
df_class.shape
```
Output:(174,9)
### Step-4: Exploratory Data Analysis
#### creating an rp analysis in percentage
```python
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
```
explanation:
df_class["Resourse Person"]: This part extracts the column named "Resourse Person" from the DataFrame df_class.

.value_counts(): This function counts the occurrences of each unique value in the specified column, which is "Resourse Person" in this case.

normalize=True: The normalize parameter is set to True, which means the counts will be normalized to represent relative frequencies (percentages) instead of absolute counts.

*100: After normalization, the counts are multiplied by 100 to convert the relative frequencies into percentages.

round(..., 2): The resulting percentages are then rounded to two decimal places.
Output:
Resourse Person
Mrs. Akshara Sasidharan    34.48
Mrs. Veena A Kumar         31.03
Dr. Anju Pratap            17.24
Mrs. Gayathri J L          17.24
Name: proportion, dtype: float64
#### creating a percentage analysis of Name-wise distribution of data
```python
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
Output:
Name
Sidharth V Menon             4.02
Rizia Sara Prabin            4.02
Aaron James Koshy            3.45
Rahul Krishnan               3.45
Allen John Manoj             3.45
Christo Joseph Sajan         3.45
Jobinjoy Ponnappal           3.45
Varsha S Panicker            3.45
Nandana A                    3.45
Anjana Vinod                 3.45
Rahul Biju                   3.45
Kevin Kizhakekuttu Thomas    3.45
Lara Marium Jacob            3.45
Abia Abraham                 3.45
Shalin Ann Thomas            3.45
Abna Ev                      3.45
Aaron Thomas Blessen         2.87
Sebin Sebastian              2.87
Sani Anna Varghese           2.87
Bhagya Sureshkumar           2.87
Jobin Tom                    2.87
Leya Kurian                  2.87
Jobin Pius                   2.30
Aiswarya Arun                2.30
Muhamed Adil                 2.30
Marianna Martin              2.30
Anaswara Biju                2.30
Mathews Reji                 1.72
MATHEWS REJI                 1.72
Riya Sara Shibu              1.72
Riya Sara Shibu              1.72
Aiswarya Arun                1.15
Sarang kj                    1.15
Muhamed Adil                 1.15
Lisbeth Ajith                1.15
Jobin Tom                    0.57
Lisbeth                      0.57
Anaswara Biju                0.57
Aaron Thomas Blessen         0.57
Lisbeth Ajith                0.57
Marianna Martin              0.57
Name: proportion, dtype: float64

### Step-5:Visualization
In this part,we are visualizing the analysed data part using graphs , pie charts etc.
```python
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Resourse Person', data=df_class)
#ax.bar_label(ax.containers[0])
plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
ax =plt.subplot(1,2,2)
ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1,0.1,0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
```
#### Create subplot with 1 row and 2 columns, selecting the first subplot
``ax = plt.subplot(1, 2, 1)``

#### Create a count plot using Seaborn
``ax = sns.countplot(x='Resourse Person', data=df_class)``

#### Set title for the first subplot
``plt.title("Faculty-wise distribution of data", fontsize=20, color='Brown', pad=20)``

#### Move to the second subplot
``ax = plt.subplot(1, 2, 2)``

#### Create a pie chart for the distribution of 'Resourse Person'
``ax = df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1, 0.1, 0.1], autopct='%1.2f%%', shadow=True)``

OUTPUT:
<img width="499" alt="Screenshot 2024-02-25 193848" src="https://github.com/lara-jacob/Python-for-ML/assets/160465136/8dc5f804-0230-4ab0-bcc4-83cbbee4e71d">

<img width="485" alt="Screenshot 2024-02-25 193910" src="https://github.com/lara-jacob/Python-for-ML/assets/160465136/9aafbdf4-8b18-421a-9913-aab179c8adf4">


### Step-5:Creating a summary of responses
 A box and whisker plot or diagram (otherwise known as a boxplot), is a graph summarising a set of data. The shape of the boxplot shows how the data is distributed and it also shows any outliers. It is a useful way to compare different sets of data as you can draw more than one boxplot per graph.

`The problem statement: A study of the segmentation of the Intel Certification course participants over satisfaction level.`

In this step we are creating box plot on various attributes and resource persons.

#### 1)creating boxplot on content quality v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
OUTPUT:
![1](https://github.com/lara-jacob/Python-for-ML/assets/160465136/49c0a540-c515-4d66-83b2-2ac1835cdea6)


#### 2)creating boxplot on Effectiveness v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
OUTPUT:
![2](https://github.com/lara-jacob/Python-for-ML/assets/160465136/7caf2001-8d9a-4297-80da-e3dbbb311a93)


#### 3)creating boxplot on Relevance v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
OUTPUT:
![3](https://github.com/lara-jacob/Python-for-ML/assets/160465136/1df5b62b-53df-452a-987f-65260a20e4ea)

#### 4)creating boxplot on Overall Organization v/s Resource person
```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
OUTPUT:
![4](https://github.com/lara-jacob/Python-for-ML/assets/160465136/0e948188-2b0c-4838-bdef-89de903027d2)

#### 5)creating boxplot on Expertise v/s Resource person
```
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Expertise'])
plt.show()
```
OUTPUT:
<img width="576" alt="5" src="https://github.com/lara-jacob/Python-for-ML/assets/160465136/72755431-a6b3-4a9a-ab4d-5400619d75d5">


#### 6)creating boxplot on Branch  v/s Content quality

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
OUTPUT:
![6](https://github.com/lara-jacob/Python-for-ML/assets/160465136/c7080787-5875-482f-9ce3-b7547d52602d)

### Step-6:Unsupervised machine learning
Using K-means Clustering to identify segmentation over student's satisfaction.
#### Finding the best value of k using elbow method
#### Elbow Method in Machine Learning

The elbow method is a technique used to determine the optimal number of clusters (k) in a clustering algorithm, such as k-means. It involves plotting the sum of squared distances (inertia) against different values of k and identifying the "elbow" point.

#### Steps:

1. **Choose a Range of k Values:**
   - Select a range of potential values for the number of clusters.

2. **Run the Clustering Algorithm:**
   - Apply the clustering algorithm (e.g., k-means) for each value of k.
   - Calculate the sum of squared distances (inertia) for each clustering configuration.

3. **Plot the Elbow Curve:**
   - Plot the values of k against the corresponding sum of squared distances.
   - Look for an "elbow" point where the rate of decrease in inertia slows down.

4. **Identify the Elbow:**
   - The optimal k is often at the point where the inertia starts decreasing more slowly, forming an elbow.

#### Interpretation:

- The elbow represents a trade-off between minimizing inertia and avoiding overfitting.
- It helps to find a balanced number of clusters for the given dataset.

Remember, while the elbow method is a useful heuristic, other factors like domain knowledge and analysis goals should also be considered in determining the final number of clusters.

```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
X=df_class[input_col].values
```
#### Initialize an empty list to store the within-cluster sum of squares
```from sklearn.cluster import KMeans
wcss = []
```

#### Try different values of k
```python

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster
```
#### plotting sws v/s k value graphs
```python
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![Elbow method](https://github.com/lara-jacob/Python-for-ML/assets/160465136/03dedf56-9778-442f-b1f6-651b324b7934)

###  gridsearch method
Another method which can be used to find the optimized value of k is gridsearch method

```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
output:
Best Parameters: {'n_clusters': 5}
Best Score: -17.904781085966768
### Step-7:Implementing K-means Clustering
K-means Clustering is a model used in unsupervised learning.Here mean values are taken into account after fixing a centroid and the process is repeated.
```python
 Perform k-means clusteringprint("Best Parameters:", best_params)
print("Best Score:", best_score)
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)#
```
output:KMeans(n_clusters=3, n_init='auto', random_state=42)
#### Extracting labels and cluster centers
Get the cluster labels and centroids
```python
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
df_class.head()
```

#### Visualizing the clustering using first two features

```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
OUTPUT:

![clustering visualizastion](https://github.com/lara-jacob/Python-for-ML/assets/160465136/aa46892a-d00e-432a-a12b-77c42dcb5f99)
### Result and Conclusion

After conducting a comprehensive feedback analysis,revealed notable some  strengths including content quality,Relevance, etc.These strengths underscore areas where the organization is excelling and can serve as benchmarks for future performance.

On the other hand, areas for improvement were also identified through the analysis. Common themes or concerns expressed in the feedback included Effectiveness,Overall Organisation etc.Addressing these areas of improvement presents an opportunity for the organization to enhance and  meet the needs and expectations of stakeholders.

In conclusion, the feedback analysis has provided a nuanced understanding of stakeholders' perceptions and experiences, highlighting both strengths and areas for improvement.Moving forward, it is essential to establish mechanisms for ongoing feedback collection and analysis to ensure continuous improvement and alignment with stakeholder expectations.





