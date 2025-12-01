import pandas as pd
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# Load the earthquake data
print("Loading earthquake data...")
df = pd.read_csv('phivolcs_earthquake_data.csv')

# Process the data
print("Processing data...")

# Convert numeric columns that might be stored as strings
numeric_cols = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with missing values in essential columns after conversion
essential_cols = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
df = df.dropna(subset=essential_cols)

# Convert magnitude to binary classification (significant vs not significant)
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)

# Limit to first 10000 rows for memory management
df = df.head(10000).copy()

# Perform DBSCAN clustering
print("Performing DBSCAN clustering...")
coords = df[['Latitude', 'Longitude']].values
clustering = DBSCAN(eps=0.05, min_samples=5).fit(coords)
df['cluster_id'] = clustering.labels_

# Create additional features
print("Creating additional features...")
df['depth_magnitude_ratio'] = df['Depth_In_Km'] / (df['Magnitude'] + 1)
df['magnitude_squared'] = df['Magnitude'] ** 2
df['depth_normalized'] = df['Depth_In_Km'] / df['Depth_In_Km'].max()
df['magnitude_depth_interaction'] = df['Magnitude'] * df['Depth_In_Km']
df['distance_from_center'] = np.sqrt((df['Latitude'] - 14.5995)**2 + (df['Longitude'] - 120.9842)**2)  # Distance from Manila

# Define features for the model
feature_columns = [
    'Latitude', 'Longitude', 'Depth_In_Km', 'cluster_id', 
    'magnitude_squared', 'depth_magnitude_ratio', 'depth_normalized', 
    'magnitude_depth_interaction', 'distance_from_center'
]

# Prepare the feature matrix X and target y
X = df[feature_columns]
y = df['is_significant']

# Fill any remaining NaN values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Save the model and feature columns
print("Saving model and feature columns...")
joblib.dump(model, 'earthquake_model_improved.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("Model training completed successfully!")
print(f"Model saved to earthquake_model_improved.pkl")
print(f"Feature columns saved to feature_columns.pkl")
print(f"Model trained on {len(X_train)} samples and tested on {len(X_test)} samples")
print(f"Number of features: {len(feature_columns)}")