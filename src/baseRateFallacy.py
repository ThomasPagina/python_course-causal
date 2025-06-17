import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class BayesClassifier:
    def __init__(self, prevalence: float, sensitivity: float, specificity: float):
        self.prevalence = prevalence
        self.sensitivity = sensitivity
        self.specificity = specificity
        self.false_positive_rate = 1 - specificity

    def bayes_theorem(self) -> float:
        p_positive = (self.sensitivity * self.prevalence +
                      self.false_positive_rate * (1 - self.prevalence))
        posterior = (self.sensitivity * self.prevalence) / p_positive
        return posterior

    def simulate_classification(self, n_samples: int, random_seed: int = None):
        if random_seed:
            np.random.seed(random_seed)

        true_labels = np.random.choice(
            [1, 0], size=n_samples, p=[self.prevalence, 1 - self.prevalence]
        )

        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            if true_labels[i] == 1:
                predictions[i] = np.random.rand() < self.sensitivity
            else:
                predictions[i] = np.random.rand() < self.false_positive_rate

        return true_labels, predictions

    def evaluate_simulation(self, true_labels, predictions):
        positives = predictions == 1
        true_positives = (true_labels == 1) & positives

        if positives.sum() == 0:
            return 0.0

        ppv = true_positives.sum() / positives.sum()
        return ppv

    def plot_confusion_matrix(self, true_labels, predictions):
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not a business letter", "Business letter"],
                    yticklabels=["Not a business letter", "Business letter"])
        plt.xlabel("Prediction")
        plt.ylabel("Truth")
        plt.title("Confusion Matrix of Classification")
        plt.show()


# Application
def main():
    prevalence = 0.03
    sensitivity = 0.9
    specificity = 0.9
    n_samples = 1000

    classifier = BayesClassifier(prevalence, sensitivity, specificity)

    bayes_result = classifier.bayes_theorem()
    print(f"Bayes-computed probability that a positive result is really a business letter: {bayes_result:.2%}")

    true_labels, predictions = classifier.simulate_classification(n_samples, random_seed=42)
    simulated_ppv = classifier.evaluate_simulation(true_labels, predictions)
    print(f"Simulation result (actual PPV): {simulated_ppv:.2%}")

    classifier.plot_confusion_matrix(true_labels, predictions)


if __name__ == "__main__":
    main()
