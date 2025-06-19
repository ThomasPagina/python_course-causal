import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 1. Configuration
# ---------------------------
np.random.seed(42)
n = 1000

# ---------------------------
# 2. Simulate Gender (0 = Male, 1 = Female)
# ---------------------------
gender = np.random.binomial(n=1, p=0.5, size=n)

# ---------------------------
# 3. Simulate Ability
# ---------------------------
ability = np.random.normal(loc=0, scale=1, size=n)

# ---------------------------
# 4. Assign Department (collider)
# ---------------------------
department_score = 0.5 * ability - 1.0 * gender + np.random.normal(0, 0.5, size=n)
department = np.where(department_score > 0, 'Engineering', 'HR')

# ---------------------------
# 5. Simulate Salary
# ---------------------------
base_salary = 50000
salary = (
    base_salary +
    10000 * ability +
    np.where(department == 'Engineering', 20000, 0) +
    np.random.normal(0, 5000, size=n)
)

# ---------------------------
# 6. Build DataFrame
# ---------------------------
df = pd.DataFrame({
    'Gender': np.where(gender == 0, 'Male', 'Female'),
    'Ability': ability,
    'Department': department,
    'Salary': salary
})

# ---------------------------
# 7. First: Department-wise plots by gender (looks fair)
# ---------------------------
plt.figure(figsize=(14, 6))

# Engineering department
plt.subplot(1, 2, 1)
eng_df = df[df['Department'] == 'Engineering']
for g in ['Male', 'Female']:
    subset = eng_df[eng_df['Gender'] == g]
    plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=g)
plt.title('Engineering: Salary vs Ability by Gender')
plt.xlabel('Ability')
plt.ylabel('Salary')
plt.legend()

# HR department
plt.subplot(1, 2, 2)
hr_df = df[df['Department'] == 'HR']
for g in ['Male', 'Female']:
    subset = hr_df[hr_df['Gender'] == g]
    plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=g)
plt.title('HR: Salary vs Ability by Gender')
plt.xlabel('Ability')
plt.ylabel('Salary')
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------
# 8. Then: Overall plots showing the gender divide
# ---------------------------
plt.figure(figsize=(14, 6))

# Plot 1: Overall salary vs ability by gender
plt.subplot(1, 2, 1)
for g in ['Male', 'Female']:
    subset = df[df['Gender'] == g]
    plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=g)
plt.title('All Departments: Salary vs Ability by Gender')
plt.xlabel('Ability')
plt.ylabel('Salary')
plt.legend()

# Plot 2: Overall salary vs ability by department
plt.subplot(1, 2, 2)
for d in ['Engineering', 'HR']:
    subset = df[df['Department'] == d]
    plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=d)
plt.title('All Individuals: Salary vs Ability by Department')
plt.xlabel('Ability')
plt.ylabel('Salary')
plt.legend()

plt.tight_layout()
plt.show()
