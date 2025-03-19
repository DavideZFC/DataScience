import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
df = pd.read_csv("datasets\AB_NYC_2019.csv")
df = df[df["price"] < 1000]
df = df.dropna(subset=["availability_365"])

print(df.info())

recent_reviews = df[df["number_of_reviews"] > 5].sort_values(by="number_of_reviews", ascending=False)
print(recent_reviews)

reviews_per_neighbourhood = df.groupby("neighbourhood_group")["number_of_reviews"].sum()
print(reviews_per_neighbourhood)

reviews_per_neighbourhood.plot.bar()
plt.show()