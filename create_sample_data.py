import pandas as pd
import numpy as np

# Create a sample earthquake dataset
np.random.seed(42)
n_samples = 1000

# Generate realistic earthquake data
data = {
    'Latitude': np.random.uniform(4.0, 21.0, n_samples),  # Philippines latitude range
    'Longitude': np.random.uniform(116.0, 127.0, n_samples),  # Philippines longitude range
    'Depth_In_Km': np.random.exponential(10, n_samples),  # Exponential for realistic depth distribution
    'Magnitude': np.random.uniform(1.0, 7.0, n_samples)  # Magnitude between 1 and 7
}

df = pd.DataFrame(data)

# Save the sample data
df.to_csv('phivolcs_earthquake_data.csv', index=False)
print("Sample earthquake data created successfully with shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())