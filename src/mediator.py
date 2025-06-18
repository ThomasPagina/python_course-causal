import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Set global styles
sns.set(style="whitegrid")

# ---------------------------
# 1. Simulate Mediation Data
# ---------------------------
def simulate_data(n=1000, seed=42):
    """
    Simulates a causal system: Treatment → Mediator → Outcome (+ direct effect).
    """
    np.random.seed(seed)

    coffee = np.random.normal(loc=0, scale=1, size=n)                     # Treatment (X)
    alertness = 0.8 * coffee + np.random.normal(0, 1, size=n)             # Mediator (M)
    productivity = (1.2 * alertness + 0.5 * coffee + np.random.normal(0, 1, size=n))  # Outcome (Y)

    return pd.DataFrame({
        'Coffee': coffee,
        'Alertness': alertness,
        'Productivity': productivity
    })

# ---------------------------
# 2. Estimate Effects
# ---------------------------
def estimate_total_effect(df):
    """Regress Outcome on Treatment to estimate the total effect."""
    model = LinearRegression().fit(df[['Coffee']], df['Productivity'])
    return model.coef_[0]

def estimate_direct_effect(df):
    """Regress Outcome on Treatment and Mediator to estimate the direct effect of Treatment."""
    model = LinearRegression().fit(df[['Coffee', 'Alertness']], df['Productivity'])
    return model.coef_[0]  # Coefficient on Coffee

def estimate_indirect_effect(df):
    """Estimate indirect effect via Mediator."""
    model_mediator = LinearRegression().fit(df[['Coffee']], df['Alertness'])
    model_outcome = LinearRegression().fit(df[['Coffee', 'Alertness']], df['Productivity'])

    a = model_mediator.coef_[0]   # Coffee → Alertness
    b = model_outcome.coef_[1]    # Alertness → Productivity

    return a * b

# ---------------------------
# 3. Visualization
# ---------------------------
def plot_relationships(df):
    """Plot the relationships between Coffee, Alertness, and Productivity."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(x='Coffee', y='Alertness', data=df, ax=axs[0])
    axs[0].set_title("Coffee → Alertness")
    
    sns.scatterplot(x='Alertness', y='Productivity', data=df, ax=axs[1])
    axs[1].set_title("Alertness → Productivity")

    sns.scatterplot(x='Coffee', y='Productivity', data=df, ax=axs[2])
    axs[2].set_title("Coffee → Productivity (Total Effect)")

    for ax in axs:
        ax.set_xlabel(ax.get_title().split("→")[0].strip())
        ax.set_ylabel(ax.get_title().split("→")[-1].strip())

    plt.tight_layout()
    plt.show()

# ---------------------------
# 4. Main Execution
# ---------------------------
def main():
    df = simulate_data()

    total = estimate_total_effect(df)
    direct = estimate_direct_effect(df)
    indirect = estimate_indirect_effect(df)

    print(f"🧮 Total Effect (Coffee → Productivity):       {total:.3f}")
    print(f"➡️ Direct Effect (not through Alertness):     {direct:.3f}")
    print(f"🔁 Indirect Effect (via Alertness):           {indirect:.3f}")
    print(f"🧩 Total ≈ Direct + Indirect:                 {direct + indirect:.3f}")

    plot_relationships(df)

if __name__ == "__main__":
    main()