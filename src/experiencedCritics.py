import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
# Kurvenanpassung (logarithmische Regression)
import statsmodels.api as sm

np.random.seed(42)
n = 30

# Erfahrung der Kritiker: 1 bis 40 Jahre
erfahrung = np.random.randint(1, 41, n)

# Zeit für die Fertigstellung der Kritik (in Stunden), abnehmender Zusammenhang
# z.B. hyperbolisch, plus Rauschen
schreibzeit = 30 / (erfahrung + 2) + np.random.normal(0, 0.5, n)

# In DataFrame
df = pd.DataFrame({
    'Erfahrung (Jahre)': erfahrung,
    'Schreibzeit (Stunden)': schreibzeit
})

# Scatterplot
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Erfahrung (Jahre)', y='Schreibzeit (Stunden)', data=df)
plt.title("Erfahrung der Kritiker vs. Schreibzeit")
plt.show()

# Pearson- und Spearman-Korrelation
r_pearson, p_pearson = pearsonr(df['Erfahrung (Jahre)'], df['Schreibzeit (Stunden)'])
r_spearman, p_spearman = spearmanr(df['Erfahrung (Jahre)'], df['Schreibzeit (Stunden)'])

print(f"Pearson: r = {r_pearson:.2f}, p = {p_pearson:.3f}")
print(f"Spearman: r = {r_spearman:.2f}, p = {p_spearman:.3f}")

# Lineare Regressionslinie (zum Vergleich)
sns.regplot(x='Erfahrung (Jahre)', y='Schreibzeit (Stunden)', data=df)
plt.title("Lineare Regression")
plt.show()



df['log_Erfahrung'] = np.log(df['Erfahrung (Jahre)'])
X = sm.add_constant(df['log_Erfahrung'])
model = sm.OLS(df['Schreibzeit (Stunden)'], X).fit()

plt.scatter(df['Erfahrung (Jahre)'], df['Schreibzeit (Stunden)'])
xp = np.linspace(1, 40, 100)
plt.plot(xp, model.predict(sm.add_constant(np.log(xp))), color='red')
plt.title("Logarithmische Regression (bessere Modellierung)")
plt.xlabel("Erfahrung (Jahre)")
plt.ylabel("Schreibzeit (Stunden)")
plt.show()

print(model.summary())
