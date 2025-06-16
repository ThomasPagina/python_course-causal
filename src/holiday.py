import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
N = 1000  # number of samples
p_holiday = 0.3
p_sick_given_holiday = 0.6
p_sick_given_no_holiday = 0.1

# Simulate data
holiday = np.random.binomial(1, p_holiday, N)
sick = np.zeros(N, dtype=int)

for i in range(N):
    if holiday[i] == 1:
        sick[i] = np.random.binomial(1, p_sick_given_holiday)
    else:
        sick[i] = np.random.binomial(1, p_sick_given_no_holiday)

X = holiday.reshape(-1, 1)
y = sick

# Fit Naive Bayes model
model = GaussianNB()
model.fit(X, y)

# Predict probabilities
proba_holiday = model.predict_proba([[1]])[0]
proba_no_holiday = model.predict_proba([[0]])[0]

print("P(Sick | Holiday):", proba_holiday[1])
print("P(Sick | No Holiday):", proba_no_holiday[1])

# Visualization: Conditional Probabilities
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=["Healthy", "Sick"], y=proba_holiday, ax=axs[0])
axs[0].set_title("P(Sickness | Holiday)")
axs[0].set_ylim(0, 1)

sns.barplot(x=["Healthy", "Sick"], y=proba_no_holiday, ax=axs[1])
axs[1].set_title("P(Sickness | No Holiday)")
axs[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# Visualization: Joint Distribution
df = pd.DataFrame({'Holiday': holiday, 'Sick': sick})
pivot = pd.crosstab(df['Holiday'], df['Sick'], normalize='index')

plt.figure(figsize=(6, 4))
sns.heatmap(pivot, annot=True, cmap='Blues', fmt=".2f")
plt.title("Joint distribution: Holiday vs Sickness")
plt.xlabel("Sick")
plt.ylabel("Holiday (0 = No, 1 = Yes)")
plt.show()
