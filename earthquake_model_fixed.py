import pandas as pd
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np


print("Loading earthquake data...")
df = pd.read_csv('phivolcs_earthquake_data.csv')

# Process the data
print("Processing data...")


numeric_cols = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


essential_cols = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
df = df.dropna(subset=essential_cols)

# Convert magnitude to binary classification (significant vs not significant)
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)


print(f"Using full dataset: {len(df)} rows") 

# Perform DBSCAN clustering
print("Performing DBSCAN clustering...")
coords = df[['Latitude', 'Longitude']].values
# n_jobs=-1 uses all processor cores to speed it up
clustering = DBSCAN(eps=0.05, min_samples=5, n_jobs=-1).fit(coords)
df['cluster_id'] = clustering.labels_


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

# Train the XGBoost model with overfitting reduction techniques
print("Training XGBoost model with overfitting prevention...")
model = XGBClassifier(
    n_estimators=100,           # Reduced from default to prevent overfitting
    max_depth=4,                # Reduced depth to limit model complexity
    learning_rate=0.05,         # Lower learning rate for more conservative learning
    subsample=0.8,              # Use 80% of samples for each tree
    colsample_bytree=0.8,       # Use 80% of features for each tree
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    min_child_weight=3,         # Minimum sum of instance weight needed in a child
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Difference (Overfitting Indicator): {abs(train_accuracy - test_accuracy):.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Save the model and feature columns
print("\nSaving model and feature columns...")
joblib.dump(model, 'earthquake_model_improved.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("Model training completed successfully!")
print(f"Model saved to earthquake_model_improved.pkl")
print(f"Feature columns saved to feature_columns.pkl")
print(f"Model trained on {len(X_train)} samples and tested on {len(X_test)} samples")
print(f"Number of features: {len(feature_columns)}")