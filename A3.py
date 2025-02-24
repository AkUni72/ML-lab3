import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    """Loads the dataset from a CSV file."""
    return pd.read_csv(file_name)

def minkowski_distance(vec1, vec2, r):
    """Computes the Minkowski distance between two vectors. """
    return np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r)

def plot_minkowski_distance(data, feature1, feature2, r_range=range(1, 11)):
    """ Plots Minkowski distance between two features for different r values."""
    distances = [minkowski_distance(data[feature1].values, data[feature2].values, r) for r in r_range]

    plt.plot(r_range, distances, marker="o")  # Plot distances for different r-values
    plt.xlabel("Minkowski r-value")  # Label x-axis
    plt.ylabel("Distance")  # Label y-axis
    plt.title(f"Minkowski Distance between {feature1} and {feature2}")  # Proper title formatting
    plt.show()  # Display plot

def main():
    """Main function to execute the program."""
    file_name = "bloodtypes.csv"  # CSV file name

    # Load dataset
    df = load_data(file_name)

    # Define features to compare
    feature1 = "O+"
    feature2 = "A+"

    # Plot Minkowski distance
    plot_minkowski_distance(df, feature1, feature2)

# Ensure script runs only when executed directly
if __name__ == "__main__":
    main()
