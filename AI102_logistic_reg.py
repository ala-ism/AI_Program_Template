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

url= "https://raw.githubusercontent.com/ala-ism/AI_Program_Template/master/Sprink_Data.csv"
df = pd.read_excel(url)


# In[4]:


df.info()


# In[5]:


df.head(5)


# In[6]:


df.tail(10)


# In[7]:


#Range dans df_r un sous tableau de df filtré sur les valeurs non nulles de la colonne Rayon
df_r = df[df.Rayon.notnull()]


# In[8]:


#Donne le contenu de pyplot.scatter(x, y1, c = 'red')la celulle de la 8ème ligne et avant dernière colonne
df.iat[7,-2]


# In[9]:


#Sous tableau regroupant les sprinkler seuls
df_spk = df[df.Hyperlien == "text"]
df_spk = df_spk[df_spk.Valeur.notnull()]
x = df_spk['Position X']
y = df_spk['Position Y']
z = df_spk['Position Z']


# In[10]:


#plot des sprinkler en x y
plt.scatter(x, y, c = 'green', marker = 's')
plt.title('Positionnement des Sprinkler')
plt.xlabel("Position en x")
plt.ylabel("Position en y")


# In[11]:


#La position du collecteur est de x=0 , z=0. Seul y défini la position du collecteur. 
dis = df_spk['Position Y'].unique()
dis.sort()


# In[12]:


df_spk_gb = df_spk.groupby("Position Y").size()


# In[13]:


# Ajout d'une colonne permettant de distinguer les zones favo et défavo
conditions = [df_spk['Position Y'] == max(df_spk['Position Y']),df_spk['Position Y'] == min(df_spk['Position Y'])]
choices = ['Défavorable', 'Favorable']
df_spk['Zone'] = np.select(conditions, choices, default='Standard')


# In[14]:


ax = sns.scatterplot(x=df_spk['Position X'], y=df_spk['Position Y'], data = df_spk, hue = 'Zone')


# In[15]:


df_favo = df_spk[df_spk['Zone'] ==  "Favorable"]
df_defavo = df_spk[df_spk['Zone'] ==  "Défavorable"]
df_std = df_spk[df_spk['Zone'] ==  "Standard"]
sns.scatterplot(x=df_favo['Position X'], y=df_favo['Position Y'], data = df_favo)
sns.scatterplot(x=df_defavo['Position X'], y=df_defavo['Position Y'], data = df_defavo)


# # Question Bonus

# In[16]:


x=df_spk[['Position X','Position Y']].values
y=df_spk['Zone'].values


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

