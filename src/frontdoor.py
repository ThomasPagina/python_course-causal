import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set global style
sns.set(style="whitegrid")

def simulate_student_data(n=1000, seed=42):
    """
    Simulate data for a causal system involving jelly bean consumption,
    sugar levels, and study focus among adult students.
    """
    np.random.seed(seed)

    lifestyle = np.random.normal(0, 1, n)  # Unobserved confounder
    jelly_beans = 0.8 * lifestyle + np.random.normal(0, 1, n)  # X
    sugar_level = 1.2 * jelly_beans + np.random.normal(0, 1, n)  # M
    study_focus = -1.5 * sugar_level + 1.0 * lifestyle + np.random.normal(0, 1, n)  # Y

    return pd.DataFrame({
        'JellyBeans': jelly_beans,
        'SugarLevel': sugar_level,
        'StudyFocus': study_focus
    })

def estimate_frontdoor_effect(df):
    """
    Estimate the causal effect of JellyBeans on StudyFocus using frontdoor adjustment.
    """
    model_m = LinearRegression().fit(df[['JellyBeans']], df['SugarLevel'])
    beta_X_to_M = model_m.coef_[0]

    model_y = LinearRegression().fit(df[['SugarLevel']], df['StudyFocus'])
    beta_M_to_Y = model_y.coef_[0]

    frontdoor_effect = beta_X_to_M * beta_M_to_Y
    return frontdoor_effect

def estimate_naive_effect(df):
    """
    Estimate the naive (confounded) effect of JellyBeans on StudyFocus.
    """
    naive_model = LinearRegression().fit(df[['JellyBeans']], df['StudyFocus'])
    return naive_model.coef_[0]

def plot_relationships(df):
    """
    Generate three scatter plots showing relationships in the causal chain.
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(x='JellyBeans', y='SugarLevel', data=df, ax=axs[0])
    axs[0].set_title("Jelly Beans → Sugar Level")
    axs[0].set_xlabel("Jelly Beans per Day")
    axs[0].set_ylabel("Blood Sugar Level")

    sns.scatterplot(x='SugarLevel', y='StudyFocus', data=df, ax=axs[1])
    axs[1].set_title("Sugar Level → Study Focus")
    axs[1].set_xlabel("Sugar Level")
    axs[1].set_ylabel("Study Focus")

    sns.scatterplot(x='JellyBeans', y='StudyFocus', data=df, ax=axs[2])
    axs[2].set_title("Jelly Beans vs Study Focus (Confounded)")
    axs[2].set_xlabel("Jelly Beans per Day")
    axs[2].set_ylabel("Study Focus")

    plt.tight_layout()
    plt.show()

def main():
    df = simulate_student_data()

    frontdoor = estimate_frontdoor_effect(df)
    naive = estimate_naive_effect(df)

    print(f"🎯 Estimated Causal Effect (Frontdoor): {frontdoor:.3f}")
    print(f"⚠️ Naive (Confounded) Estimate:        {naive:.3f}")

    plot_relationships(df)

if __name__ == "__main__":
    main()
