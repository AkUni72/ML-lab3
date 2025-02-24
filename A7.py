import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def predict_with_knn(file_path):
    df = pd.read_csv(file_path)  # Load blood type data
    features = df.iloc[:, 2:].values  # Extract features

    X, y = features[:20], np.array([0]*10 + [1]*10)  # Create features and labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Split data

    neigh = KNeighborsClassifier(n_neighbors=3)  # Initialize kNN classifier
    neigh.fit(X_train, y_train)  # Train the model

    predictions = neigh.predict(X_test)  # Make predictions
    print("Predictions:", predictions)

    test_vect = X_test[0].reshape(1, -1)  # Reshape test vector
    predicted_class = neigh.predict(test_vect)  # Predict class for test vector
    print("Predicted class for test vector:", predicted_class)

predict_with_knn('bloodtypes.csv')
