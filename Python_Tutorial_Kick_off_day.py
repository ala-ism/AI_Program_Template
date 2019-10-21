#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set(color_codes=True)

import os

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# In[34]:


url = 'https://raw.githubusercontent.com/ala-ism/AI_Program_Template/master/Tree_Classificiation.csv'
df = pd.read_csv(url)

df.head()


# In[5]:


if 'Id' in df.columns:
  df.__delitem__('Id')

df.head()

# In[]:

print("test")
print("test 3")
# In[6]:


#SUMMARY OF THE DATA SET
df.shape


# In[7]:


df.info()


# In[8]:


df['tree'].unique()

print("test test")

# In[9]:


# COMPARING THE DIFFERENT NUMERICAL COLUMNS IN THE GIVEN DATASET 

df.describe()

listOfColumns = df.columns
listOfNumericalColumns = []

for column in listOfColumns:
    if df[column].dtype == float64:
        listOfNumericalColumns.append(column)

print('listOfNumericalColumns :',listOfNumericalColumns)
spices = df['tree'].unique()
print('spices :',spices)

fig, axs = plt.subplots(nrows=len(listOfNumericalColumns),ncols=len(spices),figsize=(15,15))

for i in range(len(listOfNumericalColumns)):
    for j in range(len(spices)):  
        print(listOfNumericalColumns[i]," : ",spices[j])
        axs[i,j].boxplot(df[listOfNumericalColumns[i]][df['tree']==spices[j]])
        axs[i,j].set_title(listOfNumericalColumns[i]+""+spices[j])  


# In[10]:


#descriptions
df.describe()


# In[11]:


df.groupby('tree').size()


# In[12]:


#box and whisker plots for different numerical columns
df.plot(kind='box')


# In[13]:


#HIST PLOT OF ALL NUMERICAL COLUMNS

df.hist(figsize=(10,5))
plt.show()


# In[14]:


print("HIST PLOT OF INDIVIDUAL trees")
print(spices)

for spice in spices:
        df[df['tree']==spice].hist(figsize=(10,5))  


# In[15]:


df.boxplot(by='tree',figsize=(15,15))


# In[16]:


sns.violinplot(data=df,x='tree',y='branch.length')


# In[17]:


pd.plotting.scatter_matrix(df,figsize=(15,10))


# In[18]:


sns.pairplot(df,hue="tree")


# In[19]:


sns.pairplot(df,diag_kind='kde',hue='tree')


# ## APPLYING DIFFERENT CLASSIFICATION MODELS

# In[20]:


#Importing Metrics for Evaluation

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[21]:


# SEPARATING THE DEPENDENT AND INDEPENDENT VARIABLES ( X, Y )
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[26]:



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[27]:


from sklearn.metrics import accuracy_score

def generateClassificationReport(y_test,y_pred):
    print(classification_report(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))    
    print('accuracy is ',accuracy_score(y_test,y_pred))


# In[28]:


#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)


# In[29]:


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)


# In[30]:


#SUPPORT VECTOR MACHINE'S
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
generateClassificationReport(y_test,y_pred)


# In[31]:


#K-NEAREST NEIGHBOUR
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

generateClassificationReport(y_test,y_pred)


# In[32]:


#DECISION TREE 
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

generateClassificationReport(y_test,y_pred)

