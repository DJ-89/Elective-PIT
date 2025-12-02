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

# Further split training data to create a validation set for early stopping
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the XGBoost model with early stopping to prevent overfitting
print("Training XGBoost model with early stopping...")
model = XGBClassifier(
    n_estimators=500,           # Higher number since we'll use early stopping
    max_depth=3,                # Reduced depth to limit model complexity
    learning_rate=0.05,         # Lower learning rate for more conservative learning
    subsample=0.8,              # Use 80% of samples for each tree
    colsample_bytree=0.8,       # Use 80% of features for each tree
    reg_alpha=0.5,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    min_child_weight=3,         # Minimum sum of instance weight needed in a child
    random_state=42
)

# Note: XGBClassifier from sklearn doesn't support early stopping in fit method directly
# We'll use the native XGBoost API for early stopping
from xgboost import train as xgb_train
from xgboost import DMatrix

# Convert to DMatrix format for native API
dtrain = DMatrix(X_train_sub, label=y_train_sub)
dval = DMatrix(X_val, label=y_val)

# Define parameters
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'min_child_weight': 3,
    'random_state': 42,
    'eval_metric': 'logloss'
}

print("Training XGBoost model with early stopping...")
xgb_model = xgb_train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=20,
    verbose_eval=False
)

# Convert back to sklearn classifier for consistency
model = XGBClassifier(
    n_estimators=xgb_model.num_boosted_rounds(),
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    min_child_weight=3,
    random_state=42
)

# Re-train with optimal number of rounds
model.fit(X_train_sub, y_train_sub)

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

# Print the number of estimators used (after early stopping)
print(f"\nNumber of estimators used: {model.best_ntree_limit}")

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