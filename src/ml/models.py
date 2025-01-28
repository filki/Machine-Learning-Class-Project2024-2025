import pandas as pd

df = pd.read_csv('../../data/dataset_combined.csv')
print(df['appid'].value_counts())