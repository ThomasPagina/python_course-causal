import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Generate data (as before)
np.random.seed(42)
n = 30

image_size = np.random.uniform(2, 10, n)
num_figures = (image_size * np.random.uniform(1, 2) + np.random.normal(0, 2, n)).astype(int)
num_figures = np.clip(num_figures, 1, None)
num_assistants = (num_figures / 3 + np.random.normal(0, 0.5, n)).astype(int)
num_assistants = np.clip(num_assistants, 1, None)
honorarium = (image_size * 50 + num_figures * 30 + num_assistants * 20 + np.random.normal(0, 50, n)).astype(int)
city_size = np.random.randint(5, 50, n)

df = pd.DataFrame({
    'Image Size (m²)': image_size,
    'Number of Figures': num_figures,
    'Number of Assistants': num_assistants,
    'Honorarium (Gulden)': honorarium,
    'City Size (in 1000)': city_size
})

# 1️⃣ Regression plot: Image size vs Honorarium
plt.figure(figsize=(6, 4))
sns.regplot(x='Image Size (m²)', y='Honorarium (Gulden)', data=df)
plt.title("Image Size vs. Honorarium (with regression line)")
plt.show()

# 2️⃣ Regression plot: Number of figures vs Honorarium
plt.figure(figsize=(6, 4))
sns.regplot(x='Number of Figures', y='Honorarium (Gulden)', data=df)
plt.title("Number of Figures vs. Honorarium (with regression line)")
plt.show()

# 3️⃣ Regression plot: Number of assistants vs Honorarium
plt.figure(figsize=(6, 4))
sns.regplot(x='Number of Assistants', y='Honorarium (Gulden)', data=df)
plt.title("Number of Assistants vs. Honorarium (with regression line)")
plt.show()

# 4️⃣ Regression plot: City size vs Honorarium (here we expect no correlation)
plt.figure(figsize=(6, 4))
sns.regplot(x='City Size (in 1000)', y='Honorarium (Gulden)', data=df)
plt.title("City Size vs. Honorarium (no correlation expected)")
plt.show()

# 5️⃣ Pairplot with regression lines
sns.pairplot(df, kind='reg', diag_kind='kde')
plt.suptitle("All variables with regression lines", y=1.02)
plt.show()

# 6️⃣ Correlation matrix (for discussion)
corr = df.corr(numeric_only=True)
print("Correlation matrix (Pearson):")
print(corr)
