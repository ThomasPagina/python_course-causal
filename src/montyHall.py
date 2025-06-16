import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
N = 10000  # number of simulations

stay_wins = 0
switch_wins = 0

for _ in range(N):
    doors = [0, 1, 2]
    car = np.random.choice(doors)  # car location
    choice = np.random.choice(doors)  # player's initial choice
    
    # Monty opens a goat door
    remaining_doors = [d for d in doors if d != choice and d != car]
    monty_opens = np.random.choice(remaining_doors)
    
    # Determine remaining door after Monty's reveal
    switch_choice = [d for d in doors if d != choice and d != monty_opens][0]
    
    # Record outcomes
    if choice == car:
        stay_wins += 1
    if switch_choice == car:
        switch_wins += 1

# Compute empirical probabilities
stay_prob = stay_wins / N
switch_prob = switch_wins / N

# Print results
print(f"Stay win probability: {stay_prob:.3f}")
print(f"Switch win probability: {switch_prob:.3f}")

# Visualization
sns.barplot(x=["Stay", "Switch"], y=[stay_prob, switch_prob])
plt.title("Monty Hall Simulation Results")
plt.ylabel("Winning Probability")
plt.ylim(0, 1)
plt.show()
