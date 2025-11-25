#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd


# In[21]:


df=pd.read_csv("insurance.csv")
df.head(3)


# In[22]:


df.isnull().sum()


# In[23]:


df=df.drop(columns=["age","sex","bmi","smoker","region"])


# In[24]:


df.head(3)


# In[6]:





# In[ ]:





# In[8]:





# In[ ]:





# In[25]:


x=df.iloc[:, [0,1]].values


# In[26]:


from sklearn.cluster import KMeans


# In[15]:


pip install matplotlib


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


a=[]
for i in range(1,11):
    b=KMeans(n_clusters=i, init="k-means++", random_state=42)
    b.fit(x)
    a.append(b.inertia_)
plt.plot(range(1,11),a)

plt.title("The Elbow Method Graph")
plt.xlabel("Number of Clusters(k)")
plt.ylabel("wcss_list")
plt.show()


# In[29]:


b=KMeans(n_clusters=4, init="k-means++",random_state=42)
y_predict=b.fit_predict(x)


# In[30]:


#Visualizing the clusters
plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=100,c="blue",label="Cluster 1")
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=100,c="green",label="Cluster 2")
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=100,c="red",label="Cluster 3")
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],s=100,c="cyan",label="Cluster 4")

plt.scatter(b.cluster_centers_[:,0], b.cluster_centers_[:,1],s=300,c="yellow",label="Centroid")

plt.title("Cluster of insurance_person")
plt.xlabel("Number of Children")
plt.ylabel("Insurance Charges")
plt.legend()
plt.show()


# In[ ]:




