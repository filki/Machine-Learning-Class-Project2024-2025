import pandas as pd

# Wczytanie oryginalnego pliku CSV
df = pd.read_csv('../../data/dataset_combined.csv')

# Wybranie pierwszych 501 wierszy i zapisanie do nowego pliku
df.head(501).to_csv('../../data/test_dataset_combined.csv', index=False)