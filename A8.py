import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def plot_knn_accuracy(file_path):
    df = pd.read_csv(file_path)  # Load blood type data
    features = df.iloc[:, 2:].values  # Extract features

    X, y = features[:20], np.array([0]*10 + [1]*10)  # Create features and labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Split data

    k_values = range(1, 12)
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)  # Initialize kNN classifier
        knn.fit(X_train, y_train)  # Train the model
        accuracies.append(knn.score(X_test, y_test))  # Calculate accuracy

    plt.plot(k_values, accuracies, marker='o')  # Plot accuracies
    plt.xlabel('k value')
    plt.ylabel('Accuracy')
    plt.title('kNN Accuracy for different k values')
    plt.show()

plot_knn_accuracy('bloodtypes.csv')
