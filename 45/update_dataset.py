import pandas as pd

# Read the new dataset with upazila
df = pd.read_csv("smartgrid_risk_dataset.csv")

# Add upazila column if it doesn't exist (from the other dataset)
if "upazila" not in df.columns:
    # This is the OLD dataset - need to add upazila column
    # For now, let's just use existing columns
    pass

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)