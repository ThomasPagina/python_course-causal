import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter

def roll_dice(num_rolls=1):
    return [random.randint(1, 6) for _ in range(num_rolls)]

def get_distribution(num_rolls=1000,dices=4):
    distribution = {i: 0 for i in range(dices, dices* 6 + 1)}  # Possible sums range from 4 to 24 for 4 dice
    for _ in range(num_rolls):
        roll = sum(roll_dice(dices))

        distribution[roll] += 1
    return distribution

def plot_distribution(distribution, dices=4):
    outcomes = list(distribution.keys())
    frequencies = list(distribution.values())
    plt.bar(outcomes, frequencies, color='skyblue')
    plt.xlabel('Dice Roll Outcome')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dice Roll Outcomes')
    plt.xticks(np.arange(dices, dices*6 + 1, 1))
    plt.show()

def generate_probabilities_distribution(repeat=4):
    # all possible results (1 bis 6) for repeat dice
    all_rolls = itertools.product(range(1, 7), repeat=repeat)

    # sum of each roll
    sums = [sum(roll) for roll in all_rolls]

    # Count how often each sum occurs
    counter = Counter(sums)

    # calculate probabilities
    total = sum(counter.values())
    probabilities = {k: v / total for k, v in sorted(counter.items())}
    
    return probabilities

def probability_of_sum(sum_value, repeat=4):
    probabilities = generate_probabilities_distribution(repeat)
    return probabilities.get(sum_value, 0)

def get_first_out_of_num_rolls(value:int,num_rolls=10000, dices=4):
    for i in range(num_rolls):
        if sum(roll_dice(dices)) == value:
            return i + 1
    return num_rolls+1  # If not found, return num_rolls + 1 to indicate failure

def main():
    
    num_rolls = 10000
    distribution = get_distribution(num_rolls,4)
    expected_distribution = generate_probabilities_distribution(4)
    # multiply by num_rolls to get expected frequencies
    expected_distribution = {k: v * num_rolls for k, v in expected_distribution.items()}
    plot_distribution(distribution, 4)
    plot_distribution(expected_distribution, 4)
    print(f"Probability of rolling a sum of 24 with 4 dice: {probability_of_sum(24, 4):.4f}")
    print(f"Actual outcome of rolling a sum of 24 with 4 dices: {distribution.get(24, 0)}")
    tries = min(get_first_out_of_num_rolls(24, num_rolls, 4) for _ in range(1000))
    print(f"Earliest number of tries to roll a sum of 24 with 4 dices: {tries:d}")

if __name__ == "__main__":
    main()