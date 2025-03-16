import pandas as pd
import matplotlib.pyplot as plt

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
print(df.info())

df_by_host = df.groupby("host_id")['id'].count().sort_values(ascending=False)
print(df_by_host.head())

host_id_to_name = df.set_index("neighbourhood")["host_name"].to_dict()
print(host_id_to_name)