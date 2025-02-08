import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def feature_histogram(data, feature, bins=10):
    plt.hist(data[feature], bins=bins, edgecolor="black")
    plt.xlabel(feature," Percentage")
    plt.ylabel("Frequency")
    plt.title("Histogram of ",feature, "Distribution")
    plt.show()
    return np.mean(data[feature]), np.var(data[feature])

df = pd.read_csv("bloodtypes.csv")
mean, variance = feature_histogram(df, "O+")
print("Mean: ",mean, "Variance: ",variance)
