import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def vary_k_accuracy(X_train, X_test, y_train, y_test, k_range=range(1, 12)):
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracies.append(knn.score(X_test, y_test))
    
    plt.plot(k_range, accuracies, marker="o")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("k vs Accuracy")
    plt.show()
