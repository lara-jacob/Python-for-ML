# Unsupervised Machine Learning: Exploring Patterns in Unlabeled Data

## Introduction:
Unsupervised machine learning is a pivotal paradigm in data analysis, aimed at uncovering latent structures within unlabelled datasets. Unlike supervised learning methodologies reliant on annotated examples, unsupervised techniques autonomously identify patterns, similarities, and anomalies inherent in raw data. This technical document delves into the foundational principles, key algorithms, practical applications, and emerging trends in unsupervised machine learning.

## Foundational Concepts:
Unsupervised learning encompasses two core objectives: clustering and dimensionality reduction.
- **Clustering**: The process of partitioning data points into cohesive groups based on inherent similarities.
- **Dimensionality Reduction**: Techniques aimed at reducing the number of features while preserving the essential structure and relationships within the data.

## Algorithms in Unsupervised Learning:
1. **K-means Clustering**: An iterative algorithm that minimizes the within-cluster sum of squares by assigning data points to the nearest cluster centroid.
2. **Hierarchical Clustering**: Constructs a tree-like hierarchy of clusters by recursively merging or splitting based on distance metrics.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: A density-based algorithm capable of identifying clusters as regions of high density separated by areas of lower density, robust to noise and outliers.
4. **Principal Component Analysis (PCA)**: Linear transformation technique for dimensionality reduction, maximizing variance while projecting data onto a lower-dimensional subspace.
5. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Non-linear dimensionality reduction method preserving local similarities, commonly used for high-dimensional data visualization.
6. **Autoencoders**: Neural network architectures designed to learn efficient representations of input data, facilitating dimensionality reduction and feature learning.

## Applications of Unsupervised Learning:
1. **Clustering Applications**:
   - Market Segmentation for Targeted Marketing Campaigns
   - Image Segmentation in Medical Imaging for Tumor Detection
   - Document Clustering for Topic Modeling in Text Mining
2. **Dimensionality Reduction Applications**:
   - Visualization of High-Dimensional Data in Exploratory Data Analysis
   - Feature Extraction for Pattern Recognition in Machine Learning Models
3. **Anomaly Detection Applications**:
   - Fraud Detection in Financial Transactions
   - Intrusion Detection in Network Security
4. **Recommendation Systems**:
   - Collaborative Filtering for Personalized Product Recommendations
   - Association Rule Mining for Basket Analysis in E-commerce Platforms

## Challenges and Best Practices:
- **Evaluation Metrics**: Utilizing metrics such as silhouette score and Davies-Bouldin index for assessing clustering quality in the absence of ground truth labels.
- **Interpretability**: Employing domain knowledge and validation techniques to interpret the results of unsupervised learning models.
- **Scalability**: Addressing scalability issues through incremental learning and parallelization strategies for handling large-scale datasets.
- **Data Preprocessing**: Implementing robust preprocessing pipelines to address challenges such as missing values, feature scaling, and noise handling.

## Future Directions:
Advancements in deep learning, reinforcement learning, and unsupervised representation learning are poised to revolutionize the landscape of unsupervised machine learning. Emerging techniques such as self-supervised learning and generative models hold promise for applications in autonomous systems, healthcare informatics, and robotics.

## Conclusion:
Unsupervised machine learning serves as a cornerstone in the realm of data analytics, facilitating the extraction of actionable insights from unlabelled datasets. By leveraging clustering, dimensionality reduction, and anomaly detection methodologies, unsupervised learning empowers researchers and practitioners to unlock hidden patterns, drive data-informed decision-making, and propel innovation across diverse domains.

## Code for Heirarchical clustering

#### Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```
```
df=pd.DataFrame([10,7,28,20,35],columns=["Marks"])
```
```
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df, method='ward'))
plt.axhline(y=3, color='r', linestyle='--')
```
#### Running clustering
```
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(df)
```

# Code for KMeans Clustering

#### 1.Importing Packages and Dataset
```
#Importing Libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline`
```
```
pip install seaborn
```
###### Importing the dataset
```
#Importing the dataset
iris=pd.read_csv("iris.csv")
```
```
iris.columns
```
```
#to show first five elements
iris.head()
```
```
iris.describe()
```
```
iris.info()
```
```
# Checking the null values in the dataset
iris.isnull().sum()
```
#### 2. Exploratory Data Analysis 

###### Box plot
```
# This shows how comparision of sepal length for different species
sns.boxplot(x = 'species', y='sepal_length', data = iris)
```
```
# This shows how comparision of sepal width for different species
sns.boxplot(x = 'species', y='sepal_width', data = iris)
```
```
# This shows how comparision of petal length for different species
sns.boxplot(x = 'species', y='petal_length', data = iris)
```
```
# This shows how comparision of petal width for different species
sns.boxplot(x = 'species', y='petal_width', data = iris)
```
###### Histogram
```
## Shows distribution of the variables
iris.hist(figsize=(8,6))
plt.show()
```
###### Pairplot
```
sns.pairplot(iris, hue='species')
```
```
iris.drop(['species'],axis = 1, inplace=True)
```

###### Correlation plot
```
figsize=[10,8]
plt.figure(figsize=figsize)
sns.heatmap(iris.corr(),annot=True)
plt.show()
```
#### 3. Finding Clusters with Elbow Method
```
ssw=[]
cluster_range=range(1,10)
for i in cluster_range:
    model=KMeans(n_clusters=i,init="k-means++",n_init=10, max_iter=300, random_state=0)
    model.fit(iris)
    ssw.append(model.inertia_)
```
```
ssw_df=pd.DataFrame({"no. of clusters":cluster_range,"SSW":ssw})
print(ssw_df)
```
```
plt.figure(figsize=(12,7))
plt.plot(cluster_range, ssw, marker = "o",color="cyan")
plt.xlabel("Number of clusters")
plt.ylabel("sum squared within")
plt.title("Elbow method to find optimal number of clusters")
plt.show()
```

#### 4. Building K Means model
```
# We'll continue our analysis with n_clusters=3
kmeans=KMeans(n_clusters=3, init="k-means++", n_init=10, random_state = 42)
# Fit the model
k_model=kmeans.fit(iris)
```
```
## It returns the cluster vectors i.e. showing observations belonging which clusters 
clusters=k_model.labels_
clusters
```
```
# Importing the dataset

iris=pd.read_csv("iris.csv")
```
```
iris['clusters']=clusters
print(iris.head())
print(iris.tail())
```
```
sns.boxplot(x = 'clusters', y='petal_width', data = iris)
```
```
sns.boxplot(x = 'clusters', y='petal_length', data = iris)
```
```
## Size of each cluster
iris['clusters'].value_counts()
```
```
# Centroid of each clusters
centroid_df = pd.DataFrame(k_model.cluster_centers_, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
centroid_df
```
```
### Visualizing the cluster based on each pair of columns

sns.pairplot(iris, hue='clusters')
```

