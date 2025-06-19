import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import itertools


def generate_data(n_samples=1000, random_state=None):
    rng = np.random.default_rng(random_state)
    data = {}

    # True correlation: height and weight
    height = rng.normal(170, 10, n_samples)
    weight = height * 0.5 + rng.normal(0, 5, n_samples)
    data['height_cm'] = height
    data['weight_kg'] = weight

    # Absurd categories
    categories = {
        'hair_length_cm': ('normal', (20, 5)),
        'num_friends_starting_with_N': ('poisson', 3),
        'num_cat_images': ('poisson', 50),
        'weekly_toilet_paper_rolls': ('normal', (3, 1)),
        'hours_of_sleep': ('normal', (7, 1.5)),
        'cups_of_coffee': ('poisson', 2),
        'num_email_spam': ('poisson', 100),
        'shoe_size': ('normal', (42, 2)),
        'daily_steps': ('normal', (7000, 2000)),
        'screen_time_hours': ('normal', (4, 2)),
        'num_books_read': ('poisson', 5),
        'pets_owned': ('poisson', 1),
        'num_trips_last_year': ('poisson', 2),
        'num_coffee_mugs': ('poisson', 4),
        'height_of_houseplants': ('normal', (30, 10)),
        'num_pins_on_pinboard': ('poisson', 20),
        'gumballs_in_jar': ('poisson', 100),
        'avg_temperature_preference': ('normal', (22, 2)),
        'num_hair_bristles_lost': ('poisson', 50),
        'count_of_left_socks': ('poisson', 5),
        'games_played_last_week': ('poisson', 10),
        'number_of_google_searches': ('poisson', 20),
        'num_messages_sent': ('poisson', 200),
        'hours_spent_watching_cat_videos': ('normal', (1, 0.5)),
        'num_coins_in_pocket': ('poisson', 10),
        'times_hit_snooze_alarm': ('poisson', 2),
        # Additional absurd variables
        'length_of_nose_hair_mm': ('normal', (5, 2)),
        'num_pens_lost_per_month': ('poisson', 7),
        'hours_spent_daydreaming': ('normal', (2, 1)),
        'number_of_plant_photos': ('poisson', 15),
        'coins_collected_under_sofa': ('poisson', 12),
        'pages_in_last_read_book': ('poisson', 300),
        'seconds_until_alarm_snooze': ('poisson', 30),
        'depth_of_puddle_jumps_cm': ('normal', (10, 3)),
        'count_of_blue_socks_owned': ('poisson', 8),
        'ounces_of_coffee_poured_daily': ('normal', (16, 4)),
        'times_sneezed_during_meeting': ('poisson', 4),
        'ml_water_spilled_per_day': ('normal', (50, 20)),
        'coincidental_blinks': ('poisson', 30),
        'minutes_on_hold_phone': ('poisson', 10),
        'jokes_told_per_month': ('poisson', 8),
        'minutes_in_elevator': ('normal', (5, 2)),
        'socks_mismatched_count': ('poisson', 2),
        'ketchup_bottles_opened': ('poisson', 3),
        'pens_collected_from_floor': ('poisson', 6),
        'cookies_eaten_during_meetings': ('poisson', 5),
        'times_laughed_at_own_joke': ('poisson', 10),
        'number_of_sticky_notes_used': ('poisson', 20),
        'hours_spent_watching_tv': ('normal', (3, 1)),
        'number_of_times_yawned': ('poisson', 15),
        'times_asked_for_directions': ('poisson', 1),
        'number_of_times_said_sorry': ('poisson', 4),
        'number_of_times_asked_for_help': ('poisson', 3),
        'number_of_times_looked_at_phone': ('poisson', 50),
        'number_of_times_visited_fridge': ('poisson', 8),
        'number_of_times_asked_for_opinion': ('poisson', 6),
        'number_of_times_asked_for_favor': ('poisson', 2),
        'number_of_times_asked_for_advice': ('poisson', 5),
        'number_of_times_asked_for_recommendation': ('poisson', 4),
        'number_of_times_asked_for_feedback': ('poisson', 3),
        'number_of_times_asked_for_permission': ('poisson', 2),
        'kilometers_walked_last_week': ('normal', (10, 3)),
        'number_of_times_asked_for_clarification': ('poisson', 3),
        'sweet_treats_eaten': ('poisson', 7),
        'swimming_sessions_last_month': ('poisson', 5),
        'length_of_fingernails_mm': ('normal', (10, 2)),
        'length_of_toenails_mm': ('normal', (5, 1)),
        'width_of_eyebrows_mm': ('normal', (15, 3)),
        'pages_in_last_magazine_read': ('poisson', 50),
        'pages_in_longest_book_on_shelf': ('poisson', 800),


    }
    for name, (dist, params) in categories.items():
        if dist == 'normal':
            mu, sigma = params
            data[name] = rng.normal(mu, sigma, n_samples)
        else:
            lam = params
            data[name] = rng.poisson(lam, n_samples)

    return pd.DataFrame(data)


def benjamini_hochberg(pvals, alpha=0.05):
    """
    Perform Benjamini-Hochberg procedure and return p-value threshold.
    """
    m = len(pvals)
    sorted_indices = np.argsort(pvals)
    sorted_pvals = np.array(pvals)[sorted_indices]
    thresholds = (np.arange(1, m+1) / m) * alpha

    below = sorted_pvals <= thresholds
    if not np.any(below):
        return 0
    max_idx = np.max(np.where(below))
    return sorted_pvals[max_idx]


def find_and_plot_correlations(df, alpha=0.05, effect_size_thresh=0.2, output_dir='plots'):
    """
    Checks all feature pairs for correlation, corrects via FDR,
    filters for p <= threshold and |r| >= effect size.
    Plots scatterplots with regression line.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cols = df.columns
    results = []
    for col1, col2 in itertools.combinations(cols, 2):
        r, p = pearsonr(df[col1], df[col2])
        results.append((col1, col2, r, p))

    pvals = [p for (_, _, _, p) in results]
    m = len(pvals)
    expected_fp = alpha * m
    print(f"Expected random significances (α={alpha}): {expected_fp:.1f} of {m}")

    p_thresh = benjamini_hochberg(pvals, alpha)
    print(f"Benjamini-Hochberg p-value threshold: {p_thresh:.4f}")

    sig = [(c1, c2, r, p) for c1, c2, r, p in results
           if p <= p_thresh and abs(r) >= effect_size_thresh]
    print(f"Found correlations after FDR & |r|>={effect_size_thresh}: {len(sig)}\n")

    for col1, col2, r, p in sig:
        plt.figure()
        x, y = df[col1], df[col2]
        plt.scatter(x, y, alpha=0.5)
        m_coef, b_coef = np.polyfit(x, y, 1)
        plt.plot(x, m_coef * x + b_coef, linewidth=2)
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f'{col1} vs {col2}  (r={r:.2f}, p={p:.3f})')
        fname = f"{col1}_vs_{col2}.png".replace(' ', '_')
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    df = generate_data(n_samples=1000, random_state=None)
    find_and_plot_correlations(df)


if __name__ == '__main__':
    main()
