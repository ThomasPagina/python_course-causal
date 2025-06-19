import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import itertools

def generate_data(n_samples=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    data = {}
    # Real correlation: height and weight
    height = rng.normal(170, 10, n_samples)
    weight = height * 0.5 + rng.normal(0, 5, n_samples)
    data['height_cm'] = height
    data['weight_kg'] = weight

    # Absurd categories
    data['hair_length_cm'] = rng.normal(20, 5, n_samples)
    data['num_friends_starting_with_N'] = rng.poisson(3, n_samples)
    data['num_cat_images'] = rng.poisson(50, n_samples)
    data['weekly_toilet_paper_rolls'] = rng.normal(3, 1, n_samples)
    data['hours_of_sleep'] = rng.normal(7, 1.5, n_samples)
    data['cups_of_coffee'] = rng.poisson(2, n_samples)
    data['num_email_spam'] = rng.poisson(100, n_samples)
    data['shoe_size'] = rng.normal(42, 2, n_samples)
    data['daily_steps'] = rng.normal(7000, 2000, n_samples)
    data['screen_time_hours'] = rng.normal(4, 2, n_samples)
    data['num_books_read'] = rng.poisson(5, n_samples)
    data['pets_owned'] = rng.poisson(1, n_samples)
    data['num_trips_last_year'] = rng.poisson(2, n_samples)
    data['num_coffee_mugs'] = rng.poisson(4, n_samples)
    data['height_of_houseplants'] = rng.normal(30, 10, n_samples)
    data['num_pins_on_pinboard'] = rng.poisson(20, n_samples)
    data['gumballs_in_jar'] = rng.poisson(100, n_samples)
    data['avg_temperature_preference'] = rng.normal(22, 2, n_samples)
    data['num_hair_bristles_lost'] = rng.poisson(50, n_samples)
    data['count_of_left_socks'] = rng.poisson(5, n_samples)
    data['games_played_last_week'] = rng.poisson(10, n_samples)
    data['number_of_google_searches'] = rng.poisson(20, n_samples)
    data['num_messages_sent'] = rng.poisson(200, n_samples)
    data['hours_spent_watching_cat_videos'] = rng.normal(1, 0.5, n_samples)
    data['num_coins_in_pocket'] = rng.poisson(10, n_samples)
    data['times_hit_snooze_alarm'] = rng.poisson(2, n_samples)

    df = pd.DataFrame(data)
    return df


def find_and_plot_correlations(df, alpha=0.05, output_dir='plots'):
    """
    Checks all pairwise combinations for Pearson correlation.
    Plots scatterplots with regression line for p < alpha.
    Saves all plots in the output_dir directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cols = df.columns
    significant_pairs = []

    for col1, col2 in itertools.combinations(cols, 2):
        x = df[col1]
        y = df[col2]
        r, p = pearsonr(x, y)
        if p < alpha:
            significant_pairs.append((col1, col2, r, p))
            # Plot
            plt.figure()
            plt.scatter(x, y, alpha=0.5)
            # Regression line
            m, b = np.polyfit(x, y, 1)
            plt.plot(x, m*x + b, linewidth=2)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f'{col1} vs {col2}\nr={r:.2f}, p={p:.3f}')
            fname = f"{col1}_vs_{col2}.png".replace(' ', '_')
            plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()

    # Print summary
    print(f'Found significant correlations (alpha={alpha}): {len(significant_pairs)}')
    for col1, col2, r, p in significant_pairs:
        print(f'{col1} ↔ {col2}: r={r:.2f}, p={p:.3f}')
    print(f'Plots saved in: {output_dir}/')


def main():
    df = generate_data(n_samples=1000)
    find_and_plot_correlations(df)

if __name__ == '__main__':
    main()
