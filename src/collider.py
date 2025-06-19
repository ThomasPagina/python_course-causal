import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_books_data(n_books=1000, top_percentile=90, random_seed=42):
    """
    Simulates a dataset of books with literary quality and moral message values.
    Determines which books are included in the curriculum based on a combined score.
    """
    np.random.seed(random_seed)
    
    # Simulate two independent features
    literary_quality = np.random.normal(loc=5, scale=2, size=n_books)
    moral_message = np.random.normal(loc=5, scale=2, size=n_books)
    
    # Clip values to a 0–10 range (for interpretability)
    literary_quality = np.clip(literary_quality, 0, 10)
    moral_message = np.clip(moral_message, 0, 10)

    # Curriculum score depends positively on both features plus some random noise
    noise = np.random.normal(0, 1, n_books)
    curriculum_score = 0.6 * literary_quality + 0.6 * moral_message + noise

    # Determine which books are in the top X percentile for curriculum inclusion
    threshold = np.percentile(curriculum_score, top_percentile)
    included = curriculum_score >= threshold

    # Return all relevant data in a pandas DataFrame
    return pd.DataFrame({
        "Literary Quality": literary_quality,
        "Moral Message": moral_message,
        "Curriculum Score": curriculum_score,
        "Included in Curriculum": included
    })

def plot_collider_effect(df):
    """
    Creates four scatter plots to visualize the collider effect:
    1. Literary Quality vs Curriculum Score
    2. Moral Message vs Curriculum Score
    3. Literary Quality vs Moral Message (with inclusion coloring)
    4. Literary Quality vs Moral Message for included books only, with regression line
    """
    sns.set(style="whitegrid", context="talk")

    # Set up 1x4 subplot grid
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # Plot 1: Literary Quality vs Curriculum Score
    sns.scatterplot(
        data=df, x="Literary Quality", y="Curriculum Score",
        hue="Included in Curriculum", ax=axs[0]
    )
    axs[0].set_title("Literary Quality vs Curriculum Score")

    # Plot 2: Moral Message vs Curriculum Score
    sns.scatterplot(
        data=df, x="Moral Message", y="Curriculum Score",
        hue="Included in Curriculum", ax=axs[1]
    )
    axs[1].set_title("Moral Message vs Curriculum Score")

    # Plot 3: Literary Quality vs Moral Message (colored by inclusion)
    sns.scatterplot(
        data=df, x="Literary Quality", y="Moral Message",
        hue="Included in Curriculum", ax=axs[2]
    )
    axs[2].set_title("Literary Quality vs Moral Message")

    # Plot 4: Only included books with regression line
    included_df = df[df["Included in Curriculum"]]
    sns.regplot(
        data=included_df, x="Literary Quality", y="Moral Message",
        ax=axs[3], scatter_kws={"s": 40}
    )
    axs[3].set_title("Only Included Books (Regression)")

    # Hide legend in last plot, it's redundant
    axs[3].legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to simulate the data and produce visualizations.
    """
    df = simulate_books_data()
    plot_collider_effect(df)

# Standard Python entry point
if __name__ == "__main__":
    main()
