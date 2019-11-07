#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#%%
url="https://raw.githubusercontent.com/ala-ism/AI_Program_Template/master/Airport_Data_Rand.csv"
features=pd.read_csv(url)
features.head()
#%%
features.describe()
features=pd.get_dummies(features)
features.head()

#%%
labels=np.array(features['actual'])
features=features.drop('actual', axis=1)
features_list=list(features.columns)
features=np.array(features)
train_features, test_features, train_labels, test_labels=train_test_split(features, labels, test_size=0.25, random_state=42)

#%%
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#%%
#baseline
baseline_preds=test_features[:, features_list.index('average')]
baseline_errors=abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

#%%
rand_for=RandomForestRegressor(n_estimators=1000, random_state=42)
rand_for.fit(train_features, train_labels)

#%%
predictions=rand_for.predict(test_features)
errors=abs(predictions - test_labels)
print ('Mean Absolute Error: ', round(np.mean(errors), 2))

#%%
#accuracy and mean absolute percentage error (MAPE)
mape=100*(errors/test_labels)
accuracy=100 -np.mean(mape)
print('Accuracy:', round(accuracy, 2), "%")
