import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

df = pd.read_csv("bloodtypes.csv")
features = ["O+", "A+", "B+", "AB+"]
target = "O+"
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3) 
model = train_knn(X_train, y_train)
print("kNN model trained.")
