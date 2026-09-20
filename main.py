import pandas as pd

df = pd.read_csv("credit_risk_dataset.csv")

# print(df.info())

print(df.groupby(['person_age'])['person_income'].agg(['max', 'min', 'mean']))