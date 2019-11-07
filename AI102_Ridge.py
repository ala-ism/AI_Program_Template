#%%
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
import pandas as pd
#%%
##Deal with the data first and make sure our variables contain the right (type) of data
url="https://raw.githubusercontent.com/ala-ism/AI_Program_Template/master/Airport_Data_ridge.csv"
df=pd.read_csv(url)
df=df[~df.partition.str.contains("partition")]
df=df.dropna(subset=['acType','carrier','aldt','aibt',"stand","runway"])
df=df[["carrier","acType","stand","runway","aldt","aibt"]]
df["taxi-in"]=pd.to_datetime(df['aibt'])-pd.to_datetime(df['aldt'])
df["stand"]=df["stand"].apply(lambda x: x[5:]).astype(int)
df["runway"]=df["runway"].apply(lambda x: x[6:]).astype(int)
df["acType"]=df["acType"].apply(lambda x: x[:4])
df.head()
#%%
X=df[["stand","runway"]]
y=df[["taxi-in"]]
#%%
##Set the value of our Penalization parameter
alpha=1
#%%
##Initialize our Ridge regression
Penalized_regression=Ridge(alpha=alpha)
#%%
##Train the model
Penalized_regression.fit(X,y)
coeffiscients=Penalized_regression.coef_[0]
#%%
coeffiscients[0]
