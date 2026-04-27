import pandas as pd
df = pd.read_csv("dataset/mushroom_cleaned.xls")
df.to_csv("dataset/mushroom_cleaned.csv", index=False)
print("Done")