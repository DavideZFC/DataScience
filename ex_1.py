import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset (modifica il percorso con quello corretto)
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]

mean_prices = df.groupby("neighbourhood_group")["price"].mean()
std_prices = df.groupby("neighbourhood_group")["price"].std()

high_prices = df.groupby("neighbourhood_group")["price"].quantile(0.75)
low_prices = df.groupby("neighbourhood_group")["price"].quantile(0.25)


is_cheap = df["price"] < df["neighbourhood_group"].map(low_prices)
is_medium = (df["price"] > df["neighbourhood_group"].map(low_prices)) & (df["price"] < df["neighbourhood_group"].map(high_prices))
is_high = df["price"] > df["neighbourhood_group"].map(high_prices)

rank = 1*is_medium + 2*is_high
df['rank'] = rank

rank_to_words = {0: "cheap", 1: "medium", 2: "expensive"}
rank_name = df["rank"].map(rank_to_words)

print(rank_name)