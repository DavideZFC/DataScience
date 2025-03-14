import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset (modifica il percorso con quello corretto)
df = pd.read_csv("datasets\AB_NYC_2019.csv")

# Mostra le prime righe per verificare il caricamento
interquart_range = df['price'].quantile(0.75) - df['price'].quantile(0.25)
k = 10
upper_bound = df['price'].quantile(0.75) + k*interquart_range
lower_bound = df['price'].quantile(0.25) - k*interquart_range

df_no_price_outlier = df[(df['price'] > lower_bound) & (df['price'] < upper_bound)]

# Prezzo medio per quartiere
avg_price_neighbourhood = df_no_price_outlier.groupby(["neighbourhood_group", "room_type"])["price"].mean().sort_values(ascending=False)

# pivot table
avg_price_pivot = df.pivot_table(values="price", index="neighbourhood_group", columns="room_type", aggfunc="mean")
print(avg_price_pivot)




