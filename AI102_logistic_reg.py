#%% Load packages
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Sklearn is a package with multiple tools useful for machine learning models on Python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn import datasets

#%% Import data & preprocessing
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(data= np.c_[iris.data[:, :2], iris['target']],
                     columns= ['Grip', 'Roughness', 'Surface_type'])

df.Surface_type = df.Surface_type.map({0:'Concrete', 1:'Asphalt Old', 2:'Asphalt New'})

df.head()

#%% 

X = df[['Grip', 'Roughness']].values  # we only take the first two features.
y = df['Surface_type'].values


#%%
''' To judge of the performance of a model, we need to split our data in a training set - 
which will be used to fit the model - and a testing set - with data previously unseen by the model.
Sklearn provides tools to shuffle and split the data accordingly. 
See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#%% We create a function to visualize the main metrics 

''' In classification several metrics can be used to evaluate the quality of a predictor. 
    The function we are defining here use classification report, confusion matrices and accuracy score:
    https://en.wikipedia.org/wiki/Confusion_matrix
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html '''

def generateClassificationReport(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))    
    print('accuracy is ',accuracy_score(y_test, y_pred))


#%% CLASSIFICATION MODELS

# LOGISTIC REGRESSION 
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver="lbfgs", multi_class = 'multinomial')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
generateClassificationReport(y_test, y_pred)

# Exercise 1 : Test several parameters of the Logistic Regression function, 
# for instance the regularization coefficient C (see documentation)
# (advanced: write a for loop and print the results)





#%% DECISION TREE 
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

generateClassificationReport(y_test, y_pred)

# Exercise 2 : Test several parameters of the Decision tree
# (advanced: print one decision tree : https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)







#%% OTHER MODELS USED FOR CLASSIFICATION

#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
generateClassificationReport(y_test,y_pred)


#%% SUPPORT VECTOR MACHINE'S
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
generateClassificationReport(y_test,y_pred)


#%% K-NEAREST NEIGHBOUR
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

generateClassificationReport(y_test,y_pred)

#%%
''' SOLUTION EX 1 '''

for c in [0.001, 0.5 , 10]:
    classifier = LogisticRegression(C=c, solver="lbfgs", multi_class = 'multinomial')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    generateClassificationReport(y_test, y_pred)



# %%
''' Solution Ex 2 '''

from sklearn import tree

for max_depth in [1, 2, 3, 5]:
    clf = DecisionTreeClassifier(max_depth = max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    generateClassificationReport(y_test, y_pred)

    tree.plot_tree(clf, filled=True)
    plt.show()

