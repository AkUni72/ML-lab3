import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_name):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_name)

def split_data(data, features, target, test_size=0.3):
    """Splits data into training and testing sets."""
    return train_test_split(data[features], data[target], test_size=test_size)

def main():
    """Main function to execute the program."""
    df = load_data("bloodtypes.csv")  # Load dataset
    
    # Define feature columns and target variable
    features = ["O+", "A+", "B+", "AB+"]
    target = "O+"

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df, features, target)

    # Display the sizes of training and test sets
    print("Train set size:", len(X_train), "Test set size:", len(X_test))

# Ensure script runs only when executed directly
if __name__ == "__main__":
    main()
