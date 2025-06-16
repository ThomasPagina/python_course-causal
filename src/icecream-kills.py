import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
# ------------------------------------------------
# 4. Scheinkorrelation: Eisverkauf und Ertrinken
np.random.seed(0)
temperature = np.random.uniform(20, 40, 100)
ice_sale = temperature * 50 + np.random.normal(0, 10, 100)
drowning = temperature * 2 + np.random.normal(0, 2, 100)



print("Pearson:", pearsonr(ice_sale, drowning))

df2 = pd.DataFrame({'temperature': temperature, 'ice sale': ice_sale, 'drowning': drowning})
sns.pairplot(df2)
plt.show()
