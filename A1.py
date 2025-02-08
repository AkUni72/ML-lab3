import numpy as np
import pandas as pd

def compute_class_statistics(data, feature):
    centroid = np.mean(data[feature])
    spread = np.std(data[feature])
    return centroid, spread

def compute_interclass_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

df = pd.read_csv("bloodtypes.csv")
centroid_O, spread_O = compute_class_statistics(df, "O+")
centroid_A, spread_A = compute_class_statistics(df, "A+")
distance = compute_interclass_distance(centroid_O, centroid_A)

print("O+ Centroid: ",centroid_O, "Spread: ",spread_O)
print("A+ Centroid: ",centroid_A, "Spread: ",spread_A)
print("Interclass Distance (O+ vs A+): ",distance)
