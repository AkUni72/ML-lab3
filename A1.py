import numpy as np
import pandas as pd

def compute_class_statistics(data, feature):
    """Computes the centroid (mean) and spread (standard deviation) of a given feature."""
    centroid = np.mean(data[feature])  # Calculate mean (centroid)
    spread = np.std(data[feature])  # Calculate standard deviation (spread)
    return centroid, spread

def compute_interclass_distance(centroid1, centroid2):
    """Computes the Euclidean distance between two centroids."""
    return np.linalg.norm(centroid1 - centroid2)  # Compute Euclidean distance

def main():
    """Main function to execute the program."""
    
    # Load dataset
    file_name = "bloodtypes.csv"  # CSV file name
    df = pd.read_csv(file_name)  # Read data into a Pandas DataFrame

    # Compute statistics for blood type O+
    centroid_O, spread_O = compute_class_statistics(df, "O+")

    # Compute statistics for blood type A+
    centroid_A, spread_A = compute_class_statistics(df, "A+")

    # Compute interclass distance between O+ and A+
    distance = compute_interclass_distance(centroid_O, centroid_A)

    # Display results
    print("O+ Centroid:", centroid_O, "Spread:", spread_O)
    print("A+ Centroid:", centroid_A, "Spread:", spread_A)
    print("Interclass Distance (O+ vs A+):", distance)

# Ensure script runs only when executed directly
if __name__ == "__main__":
    main()
