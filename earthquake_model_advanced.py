import pandas as pd
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

# Hyperparameter tuning to prevent overfitting
print("Performing hyperparameter tuning...")
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0.1, 0.5, 1.0],
    'reg_lambda': [0.1, 1.0, 2.0],
    'min_child_weight': [1, 3, 5]
}

# Use a smaller subset of parameters to make grid search faster
xgb = XGBClassifier(random_state=42)

# For faster execution, using a more focused parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0.1, 0.5],
    'reg_lambda': [1.0, 1.5],
    'min_child_weight': [3, 5]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,  # Using 3-fold CV to save time
    n_jobs=-1,
    verbose=1
)

print("Fitting grid search...")
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Difference (Overfitting Indicator): {abs(train_accuracy - test_accuracy):.4f}")

# Perform cross-validation to get a more robust estimate
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

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

# Additional check for overfitting
if abs(train_accuracy - test_accuracy) > 0.1:
    print("\nWARNING: Potential overfitting detected! Difference between train and test accuracy > 0.1")
else:
    print("\nGood: Model shows minimal signs of overfitting.")