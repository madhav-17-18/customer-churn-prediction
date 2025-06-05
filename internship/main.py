from logistic import run_logistic_model
from randomforest import run_random_forest
from gradientboost import run_gradient_boosting
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Running churn prediction models...\n")

    # Collect accuracies
    results = {}
    results["Logistic Regression"] = run_logistic_model()
    results["Random Forest"] = run_random_forest()
    results["Gradient Boosting"] = run_gradient_boosting()

    print("\nAll models completed. Plotting accuracies...\n")

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()), palette="coolwarm")

    plt.title("Churn Prediction Model Accuracies", fontsize=16)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
