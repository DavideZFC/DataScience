import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
df = df.dropna()

print(df.info())
new_df = df.set_index("neighbourhood")
new_df = new_df["neighbourhood_group"].to_dict()

df["neighbourhood"] = df["neighbourhood"].map(new_df)
print(df.iloc[0,:])