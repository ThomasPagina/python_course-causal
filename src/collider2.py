import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_data(n=1000, seed=42):
    """
    Simulate gender, ability, department assignment, and salary.
    Returns a pandas DataFrame.
    """
    np.random.seed(seed)

    # Simulate binary gender (0 = Male, 1 = Female)
    gender = np.random.binomial(n=1, p=0.5, size=n)

    # Simulate ability (normally distributed)
    ability = np.random.normal(loc=0, scale=1, size=n)

    # Assign department based on gender and ability
    dept_score = 0.5 * ability - 1.0 * gender + np.random.normal(0, 0.5, size=n)
    department = np.where(dept_score > 0, 'Engineering', 'HR')

    # Simulate salary based on ability and department
    base_salary = 50000
    salary = (
        base_salary +
        10000 * ability +
        np.where(department == 'Engineering', 20000, 0) +
        np.random.normal(0, 5000, size=n)
    )

    # Build DataFrame
    df = pd.DataFrame({
        'Gender': np.where(gender == 0, 'Male', 'Female'),
        'Ability': ability,
        'Department': department,
        'Salary': salary
    })

    return df


def plot_department_wise_by_gender(df):
    """
    Plot salary vs ability within each department, split by gender.
    """
    plt.figure(figsize=(14, 6))

    # Engineering department
    plt.subplot(1, 2, 1)
    eng_df = df[df['Department'] == 'Engineering']
    for gender in ['Male', 'Female']:
        subset = eng_df[eng_df['Gender'] == gender]
        plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=gender)
    plt.title('Engineering: Salary vs Ability by Gender')
    plt.xlabel('Ability')
    plt.ylabel('Salary')
    plt.legend()

    # HR department
    plt.subplot(1, 2, 2)
    hr_df = df[df['Department'] == 'HR']
    for gender in ['Male', 'Female']:
        subset = hr_df[hr_df['Gender'] == gender]
        plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=gender)
    plt.title('HR: Salary vs Ability by Gender')
    plt.xlabel('Ability')
    plt.ylabel('Salary')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_overall_by_gender_and_department(df):
    """
    Plot salary vs ability for all individuals by gender and by department.
    """
    plt.figure(figsize=(14, 6))

    # Overall by gender
    plt.subplot(1, 2, 1)
    for gender in ['Male', 'Female']:
        subset = df[df['Gender'] == gender]
        plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=gender)
    plt.title('All Departments: Salary vs Ability by Gender')
    plt.xlabel('Ability')
    plt.ylabel('Salary')
    plt.legend()

    # Overall by department
    plt.subplot(1, 2, 2)
    for dept in ['Engineering', 'HR']:
        subset = df[df['Department'] == dept]
        plt.scatter(subset['Ability'], subset['Salary'], alpha=0.6, label=dept)
    plt.title('All Individuals: Salary vs Ability by Department')
    plt.xlabel('Ability')
    plt.ylabel('Salary')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    df = simulate_data()
    plot_department_wise_by_gender(df)     # Step 1: Looks fair within departments
    plot_overall_by_gender_and_department(df)  # Step 2: Bias revealed across departments


if __name__ == "__main__":
    main()
