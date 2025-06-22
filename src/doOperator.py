import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def simulate_data(n=1000, seed=42):
    np.random.seed(seed)
    confounder = np.random.normal(0, 1, n)
    education = confounder + np.random.normal(0, 1, n)
    income = 2 * education + 3 * confounder + np.random.normal(0, 1, n)

    return pd.DataFrame({
        'confounder': confounder,
        'education': education,
        'income': income
    })


def estimate_adjusted_model(data):
    X = data[['education', 'confounder']]
    y = data['income']
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict_do_intervention(model, education_range, confounder_value):
    confounder_array = np.full_like(education_range, confounder_value)
    X_new = np.column_stack((education_range, confounder_array))
    return model.predict(X_new)


def plot_results(data, education_range, do_income):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['education'], data['income'], alpha=0.3, label="Observed data")
    plt.plot(education_range, do_income, color='red', linewidth=2, label=r"Estimated do($education$)")
    plt.xlabel("Education")
    plt.ylabel("Income")
    plt.title("Observed vs Intervened (do-operator) Relationship")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    data = simulate_data()
    model = estimate_adjusted_model(data)

    education_range = np.linspace(-2, 4, 100)
    mean_confounder = np.mean(data['confounder'])

    do_income = predict_do_intervention(model, education_range, mean_confounder)

    plot_results(data, education_range, do_income)


if __name__ == "__main__":
    main()
