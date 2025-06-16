import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

np.random.seed(42)
n = 30  # Sample size

# Area between 2 and 10 m²
painting_size = np.random.uniform(2, 10, n)

# Number of figures: depends on area, with variation
num_figures = (painting_size * np.random.uniform(1, 2) + np.random.normal(0, 2, n)).astype(int)
num_figures = np.clip(num_figures, 1, None)

# Number of employees: depends on number of figures, with variation
num_employees = (num_figures / 3 + np.random.normal(0, 0.5, n)).astype(int)
num_employees = np.clip(num_employees, 1, None)

# Fee: depends on area, figures, and employees
fee = (painting_size * 50 + num_figures * 30 + num_employees * 20 + np.random.normal(0, 50, n)).astype(int)

# Independent additional variable: city size of the client (e.g. population in thousands)
city_size = np.random.randint(5, 50, n)  # small to medium-sized cities, purely random

# DataFrame
df = pd.DataFrame({
    'Painting Size (m²)': painting_size,
    'Number of Figures': num_figures,
    'Number of Employees': num_employees,
    'Fee (Gulden)': fee,
    'City Size (in 1000)': city_size
})

print(df.head())

# Visualization
sns.pairplot(df)
plt.show()

# Pearson correlation
corr = df.corr(numeric_only=True)
print("Correlation matrix (Pearson):")
print(corr)
