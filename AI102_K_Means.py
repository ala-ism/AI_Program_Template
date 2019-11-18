#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np

#%%
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
