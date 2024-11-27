import pandas as pd
import glob
import os

csv_files = glob.glob(os.path.join(os.path.dirname(__file__), '../../data/*.csv'))

print(f"Found {len(csv_files)} CSV files. Combining them into 'dataset_combined.csv'...")

# Read and combine all CSV files
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Save the combined DataFrame to a single CSV file
combined_df.to_csv('dataset_combined.csv', index=False, encoding='utf-8')

print("Successfully combined CSVs into 'dataset_combined.csv'")