import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

np.random.seed(42)
n = 30

# Kritikschärfe: 0 = sehr mild, 10 = extrem scharf
schaerfe = np.random.uniform(0, 10, n)

# Publikumsinteresse: Umgekehrte U-Kurve
interesse = -1 * (schaerfe - 5)**2 + 25 + np.random.normal(0, 2, n)

df = pd.DataFrame({
    'Kritikschärfe': schaerfe,
    'Publikumsinteresse': interesse
})

# Scatterplot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Kritikschärfe', y='Publikumsinteresse', data=df)
plt.title("Umgekehrte U-Kurve: Schärfe vs. Interesse")
plt.show()

# Pearson- und Spearman-Korrelation
r_pearson, p_pearson = pearsonr(df['Kritikschärfe'], df['Publikumsinteresse'])
r_spearman, p_spearman = spearmanr(df['Kritikschärfe'], df['Publikumsinteresse'])

print(f"Pearson: r = {r_pearson:.2f}, p = {p_pearson:.3f}")
print(f"Spearman: r = {r_spearman:.2f}, p = {p_spearman:.3f}")

# Lineare Regression demonstrativ
sns.regplot(x='Kritikschärfe', y='Publikumsinteresse', data=df)
plt.title("Lineare Regression (ungeeignet)")
plt.show()
