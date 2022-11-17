import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df= pd.read_csv("sales_data_sample.csv", encoding="latin1")
df.info()
X= df.iloc[:, [1,4]].values
# Elbow Method
wcss_list= []
for i in range(1, 11):
    kmeans = KMeans (n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(X)
    wcss_list.append(kmeans. inertia_)
    
plt.plot(range(1, 11), wcss_list)
plt.title( 'Elbow Method Graph')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('wcss_list')
plt.show()

# K-Means Clustering
kmeans= KMeans (n_clusters=3, init='k-means++', random_state= 42)
y_pred = kmeans. fit_predict(X)

plt.scatter(X[y_pred==0,0], X[y_pred==0,1],c='blue',label='Cluster 1')
plt.scatter(X[y_pred==1,0], X[y_pred==1,1],c='green',label='Cluster 2')
plt.scatter(X[y_pred==2,0], X[y_pred==2,1],c='red',label='Cluster 3')
plt.title('K-Means Clustering') 
plt.xlabel('Quantity Ordered')
plt.ylabel("Sales")
plt.legend()
plt.show()