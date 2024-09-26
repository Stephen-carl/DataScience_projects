import pickle
import pandas as pd

# Data wrangling
def wrangle(filename):
    # read the file
    df = pd.read_csv(filename).set_index("company_name")

    return df

def make_predictions(data_filepath, model_filepath):
    # Wrangle JSON file
    X_test = wrangle(data_filepath)
    # Load model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    # Generate predictions
    y_test_pred = model.predict(X_test)
    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_test_pred = pd.Series(y_test_pred, index = X_test.index, name = "bankrupt")
    return y_test_pred