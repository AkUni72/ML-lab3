import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def knn_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)

df = pd.read_csv("bloodtypes.csv")
features = ["O+", "A+", "B+", "AB+"]
target = "O+"
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
    
accuracy = knn_accuracy(knn, X_test, y_test)
print("Model accuracy: ",accuracy)