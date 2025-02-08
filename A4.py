import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(data, features, target, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=test_size)
    return X_train, X_test, y_train, y_test

df = pd.read_csv("bloodtypes.csv")
features = ["O+", "A+", "B+", "AB+"]
target = "O+"
X_train, X_test, y_train, y_test = split_data(df, features, target)
print("Train set size: ",len(X_train), "Test set size: ",len(X_test))
