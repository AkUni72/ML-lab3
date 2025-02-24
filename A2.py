import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_name):
    """ Loads the dataset from a CSV file."""
    return pd.read_csv(file_name)

def feature_histogram(data, feature, bins=10):
    """Plots a histogram for a given feature and returns its mean and variance."""
    plt.hist(data[feature], bins=bins, edgecolor="black")  # Create histogram
    plt.xlabel(f"{feature} Percentage")  # Corrected xlabel formatting
    plt.ylabel("Frequency")  # Label y-axis
    plt.title(f"Histogram of {feature} Distribution")  # Corrected title formatting
    plt.show()  # Display the histogram

    # Compute and return mean and variance of the feature
    return np.mean(data[feature]), np.var(data[feature])

def main():
    """Main function to execute the program."""
    file_name = "bloodtypes.csv"  # CSV file name

    # Load dataset
    df = load_data(file_name)

    # Define feature column for analysis
    feature_name = "O+"  

    # Generate histogram and compute statistics
    mean, variance = feature_histogram(df, feature_name)

    # Display computed mean and variance
    print(f"Mean of {feature_name}: {mean}, Variance: {variance}")

# Ensure script runs only when executed directly
if __name__ == "__main__":
    main()
