import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def knn_predict(model, X_test):
    return model.predict(X_test)

df = pd.read_csv("bloodtypes.csv")
features = ["O+", "A+", "B+", "AB+"]
target = "O+"
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

predictions = knn_predict(knn, X_test)
print(predictions)