# ANALYSING STUDENT'S BEHAVIOUR

## Importing necessary libraries
The required libraries such as pandas,numpy,seaborn,mathplotlib etc are imported.

NB:Install the libraries if not present,using pip install corresponding library name.

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```

## Loading the data to be analysed
The dataset can be in the form of csv file or may be directly from another sites.Specify the path of the dataset that we need to upload.

```
df_class=pd.read_csv("Student behaviour.csv")
df_class.head()
```

## Optimizing the dataset

In pandas, the info() method is used to get a concise summary of a DataFrame. This method provides information about the DataFrame, including the data types of each column, the number of non-null values, and memory usage. It's a handy tool for quickly assessing the structure and content of your DataFrame.
```
df_class.info()
```
removing unnecessary columns
```
df_class = df_class.drop(['NationalITy','PlaceofBirth','StageID','GradeID','SectionID','Topic','Semester','Relation'],axis=1)
```
## Exploratory Data Analysis
Relation between raising hands and classification
```
df_class['raisedhands'] = pd.cut(df_class.raisedhands, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['raisedhands'])['Class'].value_counts(normalize=True)
```
OUTPUT:
raisedhands  Class
0            L        0.534314
             M        0.392157
             H        0.073529
1            M        0.577778
             H        0.288889
             L        0.133333
2            H        0.543011
             M        0.424731
             L        0.032258
Name: proportion, dtype: float64

Relation between visited resourses and classification

```
df_class['VisITedResources'] = pd.cut(df_class.VisITedResources, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['VisITedResources'])['Class'].value_counts(normalize=True)
```
OUTPUT:
VisITedResources  Class
0                 L        0.656250
                  M        0.293750
                  H        0.050000
1                 M        0.560976
                  H        0.231707
                  L        0.207317
2                 M        0.495798
                  H        0.483193
                  L        0.021008
Name: proportion, dtype: float64

Relation between Announcements View  and classification
```
df_class['AnnouncementsView'] = pd.cut(df_class.AnnouncementsView, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['AnnouncementsView'])['Class'].value_counts(normalize=True)
```
OUTPUT:
AnnouncementsView  Class
0                  L        0.468354
                   M        0.388186
                   H        0.143460
1                  M        0.506667
                   H        0.393333
                   L        0.100000
2                  H        0.526882
                   M        0.462366
                   L        0.010753
Name: proportion, dtype: float64

Relation between Discussion  and classification
```
df_class['Discussion'] = pd.cut(df_class.Discussion, bins=3, labels=np.arange(3), right=False)
df_class.groupby(['Discussion'])['Class'].value_counts(normalize=True)
```
OUTPUT:
iscussion  Class
0           M        0.416290
            L        0.371041
            H        0.212670
1           M        0.538462
            H        0.253846
            L        0.207692
2           H        0.480620
            M        0.379845
            L        0.139535
Name: proportion, dtype: float64

Relation between StudentAbsenceDays and classification

```
df_class.groupby(['StudentAbsenceDays'])['Class'].value_counts(normalize=True)
```
OUTPUT:
tudentAbsenceDays  Class
Above-7             L        0.607330
                    M        0.371728
                    H        0.020942
Under-7             M        0.484429
                    H        0.477509
                    L        0.038062
Name: proportion, dtype: float64

## Visualization

1)creating boxplot on raisedhands v/s Class
```
sns.boxplot(y=df_class['Class'],x=df_class['raisedhands'])
plt.show()
```
OUTPUT:

2)creating boxplot on VisITedResources v/s Class
```
sns.boxplot(y=df_class['Class'],x=df_class['VisITedResources'])
plt.show()
```
OUTPUT:

3)creating boxplot on AnnouncementsView v/s Class
```
sns.boxplot(y=df_class['Class'],x=df_class['AnnouncementsView'])
plt.show()
```
OUTPUT:


4)creating boxplot on Discussion v/s Class
```
sns.boxplot(y=df_class['Class'],x=df_class['Discussion'])
plt.show()
```
OUTPUT:

5)creating boxplot on StudentAbsenceDays v/s Class
```
sns.boxplot(y=df_class['Class'],x=df_class['StudentAbsenceDays'])
plt.show()
```
## Correlation 
```
correlation = df_class[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].corr(method='pearson')
correlation
```
OUTPUT:

## Elbow Method

```
X = df_class[['raisedhands', 'VisITedResources']].values
wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    #print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss,marker='o')
plt.title('Elbow Method')
plt.xlabel('N of Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()
```

## k-Means Clustering
```
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)
```

```
k_means_labels = kmeans.labels_
k_means_cluster_centers = kmeans.cluster_centers_
```

```
plt.scatter(X[:, 0], X[:,1], s = 10, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'red',label = 'Centroids')
plt.title('Students Clustering')
plt.xlabel('RaisedHands')
plt.ylabel('VisITedResources')
plt.legend()

plt.show()
```
OUTPUT:

