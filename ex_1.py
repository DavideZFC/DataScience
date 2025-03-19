import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]


pivo = df.pivot_table(values="price", index="host_id", columns="neighbourhood_group", aggfunc="mean")

pivo.fillna(row.mean())
host_sd = pivo.std(axis=1)

# Create a DataFrame from the result
host_sd_df = host_sd.reset_index().rename(columns={0: "host sd", "host_id": "host_id"})

# Display result
print(host_sd)