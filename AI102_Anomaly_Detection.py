#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from sklearn.ensemble import IsolationForest
# %%
#generating data
rng = np.random.RandomState(42)

# Generating training data 
X_train = 0.2 * rng.randn(1000, 2)
X_train = np.r_[X_train + 3, X_train]
X_train = pd.DataFrame(X_train, columns = ['x1', 'x2'])

# Generating new, 'normal' observation
X_test = 0.2 * rng.randn(200, 2)
X_test = np.r_[X_test + 3, X_test]
X_test = pd.DataFrame(X_test, columns = ['x1', 'x2'])

# Generating outliers
X_outliers = rng.uniform(low=-1, high=5, size=(50, 2))
X_outliers = pd.DataFrame(X_outliers, columns = ['x1', 'x2'])

#%%
g1=(X_outliers['x1'],X_outliers['x2'])
g2=(X_test['x1'],X_test['x2'])
g3=(X_train['x1'],X_train['x2'])
data=(g1,g2)#,g3)
groups=("outliers","normal")#,"training")
colors=('red','white')#,'blue')
fig=plt.figure()
ax=fig.add_subplot(1,1,1)

for data, color, group in zip(data, colors, groups):
    x, y=data
    ax.scatter(x,y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('outlier detection result')
plt.legend(loc=2)
plt.show()