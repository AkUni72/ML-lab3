import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def minkowski_distance(vec1, vec2, r):
    return np.sum(np.abs(vec1 - vec2) ** r) ** (1/r)

def plot_minkowski_distance(data, feature1, feature2, r_range=range(1, 11)):
    distances = [minkowski_distance(data[feature1], data[feature2], r) for r in r_range]
    plt.plot(r_range, distances, marker="o")
    plt.xlabel("Minkowski r-value")
    plt.ylabel("Distance")
    plt.title("Minkowski Distance between ",feature1 ,"and" ,feature2)
    plt.show()

df = pd.read_csv("bloodtypes.csv")
plot_minkowski_distance(df, "O+", "A+")
