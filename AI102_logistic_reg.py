#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:

url= "https://raw.githubusercontent.com/ala-ism/AI_Program_Template/master/Sprinklers.xlsx"
df = pd.read_excel(url)

#%%
x=df_spk[['Position X','Position Y']].values
y=df_spk['Zone'].values
df_spk.to_csv("file.csv",sep="\t",encoding='utf-8')

# In[17]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.7, random_state=0)


# In[18]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[19]:


def generateClassificationReport(y_test,y_pred):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))    
    print('accuracy is ',accuracy_score(y_test,y_pred))


# In[21]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver="newton-cg")
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)


# In[22]:


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)


# In[23]:


#SUPPORT VECTOR MACHINE'S
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)


# In[24]:


#K-NEAREST NEIGHBOUR
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

generateClassificationReport(y_test,y_pred)


# In[25]:


#DECISION TREE 
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

generateClassificationReport(y_test,y_pred)

