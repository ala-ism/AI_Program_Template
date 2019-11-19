#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px


#%%
dataset = pd.read_csv('Sprinkler_data.csv',index_col=0)
dataset.head()

#%%
#We remove the column 'Zone' which is the column we are trying to predict
dataset.drop(['Zone'],axis=1,inplace=True)
dataset.head()

#%%
#We start by implementing 'The elbow method' on our dataset. 
#The elbow method allows us to pick the optimum number of clusters for classificationby visualizing the error for each number of cluster
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(dataset)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()

#%%
#Creating the kmeans classifier see documentation : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#Change the parameters to see how it changes the clusters (change the visualization too)

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
dataset['prediction'] = kmeans.fit_predict(dataset)
dataset.head()

#%%
#Visualising the clusters

fig = px.scatter_3d(dataset, x='Height', y='Flow', z='Distance_water_source_x',
              color='prediction')
fig.show()




#%%
#2nd Example with image


img=cv2.imread(r"C:/Users/test.jpg")
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

#We convert the picture from a 1666x2235 matrix to a vector
vectorized = img.reshape((-1,3))
print(vectorized.shape)
vectorized = np.float32(vectorized)


#%%
#Creation of the K-means model see documentation : https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
#try playing with the K parameter and see what happens (e.g., K=2, K=3)
K = 5
attempts=10

#Specify iteration termination criteria for K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

#Visualization of the result
res = center[label.flatten()]
result_image = res.reshape((img.shape))
figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()



# %%
