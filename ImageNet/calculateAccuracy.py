import pandas as pd

results = pd.read_csv("results.csv")

for i in range(3):
    subset = results[results["actual"]==i]
    accuracy = len(subset[subset["prediction"]==i])/len(subset) * 100
    print(f"Accuracy for Category #{i}: {accuracy:.2f}%")
