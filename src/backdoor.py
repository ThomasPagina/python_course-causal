import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set plot style
sns.set(style="whitegrid")

def simulate_data(n=1000, seed=42):
    """Generate simulated data with a backdoor path: Age → Exercise, Age → Health."""
    np.random.seed(seed)
    
    age = np.random.normal(loc=50, scale=10, size=n)
    exercise = -0.3 * age + np.random.normal(loc=0, scale=5, size=n)
    health = 0.5 * exercise - 0.4 * age + np.random.normal(loc=0, scale=5, size=n)

    return pd.DataFrame({'Age': age, 'Exercise': exercise, 'Health': health})

def plot_relationships(df):
    """Plot raw relationships between Age, Exercise, and Health."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(x='Exercise', y='Health', data=df, ax=axs[0])
    axs[0].set_title("Exercise vs Health (Raw)")
    axs[0].set_xlabel("Exercise")
    axs[0].set_ylabel("Health")

    sns.scatterplot(x='Age', y='Exercise', data=df, ax=axs[1])
    axs[1].set_title("Age vs Exercise")
    axs[1].set_xlabel("Age")
    axs[1].set_ylabel("Exercise")

    sns.scatterplot(x='Age', y='Health', data=df, ax=axs[2])
    axs[2].set_title("Age vs Health")
    axs[2].set_xlabel("Age")
    axs[2].set_ylabel("Health")

    plt.tight_layout()
    plt.show()

def regress_out(df, predictor, target):
    """Regress out the predictor from the target and return residuals."""
    model = LinearRegression().fit(df[[predictor]], df[target])
    residuals = df[target] - model.predict(df[[predictor]])
    return residuals

def plot_residual_relationship(x_resid, y_resid):
    """Plot residuals to visualize the effect of controlling for a confounder."""
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=x_resid, y=y_resid)
    plt.title("Exercise vs Health (Controlling for Age)")
    plt.xlabel("Exercise Residuals")
    plt.ylabel("Health Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = simulate_data()

    # Plot raw data relationships
    plot_relationships(df)

    # Correlation before controlling for Age
    raw_corr = df[['Exercise', 'Health']].corr().iloc[0, 1]
    print(f"Raw correlation (Exercise vs Health): {raw_corr:.3f}")

    # Regress out Age from Exercise and Health
    exercise_resid = regress_out(df, predictor='Age', target='Exercise')
    health_resid = regress_out(df, predictor='Age', target='Health')

    # Plot controlled relationship
    plot_residual_relationship(exercise_resid, health_resid)

    # Partial correlation
    partial_corr = np.corrcoef(exercise_resid, health_resid)[0, 1]
    print(f"Partial correlation (controlling for Age): {partial_corr:.3f}")

if __name__ == "__main__":
    main()
