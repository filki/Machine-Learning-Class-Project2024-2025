# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required to enable MICE in sklearn
from sklearn.impute import IterativeImputer

# Step 1: Generate Synthetic Dataset with Missing Values
np.random.seed(42)
data = {
    "A": np.random.choice([1, 2, np.nan, 4, 5], 100),
    "B": np.random.choice([10, np.nan, 30, 40, 50], 100),
    "C": np.random.choice([100, 200, 300, np.nan], 100),
    "D": np.random.randn(100)
}
df_synthetic = pd.DataFrame(data)
print("Original Dataset with Missing Values:")
print(df_synthetic.head())

# Step 2: Initialize the Iterative Imputer (MICE)
imputer = IterativeImputer(max_iter=10, random_state=0)

# Step 3: Apply MICE Imputation
df_imputed = pd.DataFrame(imputer.fit_transform(df_synthetic), columns=df_synthetic.columns)

# Display the imputed dataset
print("\nImputed Dataset:")
print(df_imputed.head())
