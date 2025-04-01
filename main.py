import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utilities import HOUSING_PATH
# from utilities import fetch_housing_data
from sklearn.model_selection import train_test_split

# Run once to download the data
# fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)    

housing = load_housing_data()
print(housing.head())

# Split the data into training and test sets
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Training set size: {len(train_set)}\nTest set size: {len(test_set)}")

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])  
housing["income_cat"].hist()
plt.show()