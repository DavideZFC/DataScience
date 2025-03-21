import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]


multi_host = df.groupby("host_id")["neighbourhood_group"].nunique() >= 3
df["multihost"] = df["host_id"].map(multi_host)

df_multihost = df[df["multihost"]]
neig_series = df_multihost.groupby("host_id")["neighbourhood_group"].nunique()
print(neig_series)

pivot_price_host_nei = df_multihost.pivot_table(values="price", index="host_id", columns="neighbourhood_group", aggfunc="mean")
std_series = pivot_price_host_nei.std(axis=1)
print(std_series)

pivot_count_host_room = df_multihost.pivot_table(values="price", index="host_id", columns="room_type", aggfunc="count")
print(pivot_count_host_room)

room_series = df_multihost.groupby("host_id")["room_type"].nunique()
print(room_series)

new_data = pd.DataFrame({"num_neighbourhoods": neig_series, "price_std": std_series, "num_room_types": room_series})
print(new_data)

listing_count = df_multihost.groupby("host_id")["id"].count()
new_data["num_listings"] = listing_count

plt.figure(figsize=(8, 5))
sns.scatterplot(data=new_data, x="num_room_types", y="price_std", size="num_listings", alpha=0.7)
plt.title("Variabilità di Prezzo vs. Diversità di Alloggi")
plt.xlabel("Numero di Tipi di Alloggio")
plt.ylabel("Deviazione Std del Prezzo tra Quartieri")
plt.show()