import pandas as pd
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data in chunks to manage memory
print("Loading earthquake data...")
df = pd.read_csv('phivolcs_earthquake_data.csv', nrows=10000)  # Limit to first 10k rows for memory management

# 2. Data Cleaning (Updated to fix crash)
print("Cleaning data...")
# Convert Date
df['Date_Time_PH'] = pd.to_datetime(df['Date_Time_PH'], errors='coerce')

# Force numeric columns (fixes errors like '<001' or 'Unknown')
cols_to_clean = ['Latitude', 'Longitude', 'Depth_In_Km', 'Magnitude']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any rows that still have missing values
df = df.dropna(subset=cols_to_clean)

# Remove future dates (beyond current date)
current_date = datetime.now()
df = df[df['Date_Time_PH'] <= current_date]

print(f"Dataset shape after cleaning: {df.shape}")

# 3. Create Target & Cluster
df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)

# Extract temporal features
df['Year'] = df['Date_Time_PH'].dt.year
df['Month'] = df['Date_Time_PH'].dt.month
df['Day'] = df['Date_Time_PH'].dt.day
df['Hour'] = df['Date_Time_PH'].dt.hour
df['DayOfYear'] = df['Date_Time_PH'].dt.dayofyear

# Parse location into regions (Luzon, Visayas, Mindanao)
def classify_region(lat, lon):
    if 5 <= lat <= 18 and 116 <= lon <= 127:  # Philippines general area
        if lat >= 15:  # Northern Philippines
            return 'Luzon'
        elif 8 <= lat < 15:  # Central Philippines
            return 'Visayas'
        else:  # Southern Philippines
            return 'Mindanao'
    return 'Unknown'

df['Region'] = df.apply(lambda row: classify_region(row['Latitude'], row['Longitude']), axis=1)

# Create dummy variables for regions
region_dummies = pd.get_dummies(df['Region'], prefix='Region')
df = pd.concat([df, region_dummies], axis=1)

# 4. Enhanced Feature Engineering for Clustering
print("Performing DBSCAN clustering...")
coords = df[['Latitude', 'Longitude']]

# DBSCAN Clustering (Hotspots) - More conservative parameters to avoid overfitting
# eps=0.1 (~11km), min_samples=10 - more conservative clustering
db = DBSCAN(eps=0.1, min_samples=10).fit(coords)
df['cluster_id'] = db.labels_

# Add cluster statistics as features
cluster_stats = df.groupby('cluster_id').agg({
    'Magnitude': ['mean', 'std', 'count'],
    'Depth_In_Km': ['mean', 'std']
}).fillna(0)

cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
cluster_stats = cluster_stats.add_prefix('cluster_')

df = df.merge(cluster_stats, left_on='cluster_id', right_index=True, how='left')

# 5. Additional Feature Engineering
print("Creating additional features...")
# Depth-Magnitude ratio
df['depth_magnitude_ratio'] = df['Depth_In_Km'] / (df['Magnitude'] + 0.001)  # Adding small value to avoid division by zero

# Magnitude squared (to capture non-linear effects)
df['magnitude_squared'] = df['Magnitude'] ** 2

# Normalized depth
df['depth_normalized'] = (df['Depth_In_Km'] - df['Depth_In_Km'].mean()) / df['Depth_In_Km'].std()

# Spatial interaction features
df['lat_long_interaction'] = df['Latitude'] * df['Longitude']

# Distance from center of Philippines (approximate)
ph_center_lat, ph_center_lon = 12.8797, 121.7740
df['distance_from_center'] = np.sqrt((df['Latitude'] - ph_center_lat)**2 + (df['Longitude'] - ph_center_lon)**2)

# 6. Define Inputs (X) and Target (y)
feature_cols = [
    'Latitude', 'Longitude', 'Depth_In_Km', 'cluster_id',
    'Year', 'Month', 'Day', 'Hour', 'DayOfYear',
    'Region_Luzon', 'Region_Mindanao', 'Region_Visayas', 'Region_Unknown',
    'cluster_Magnitude_mean', 'cluster_Magnitude_std', 'cluster_Magnitude_count',
    'cluster_Depth_In_Km_mean', 'cluster_Depth_In_Km_std',
    'depth_magnitude_ratio', 'magnitude_squared', 'depth_normalized',
    'lat_long_interaction', 'distance_from_center'
]

X = df[feature_cols]
y = df['is_significant']

# Fill NaN values that might have been created
X = X.fillna(X.mean())

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# 7. SPLIT with Stratification and Shuffling - More realistic split
print("Splitting data...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print(f"Training on {len(X_train)} events (80%)")
print(f"Testing on {len(X_test)} events (20%)")

# 8. Train XGBoost with More Conservative Parameters to Prevent Overfitting
print("Training XGBoost model with conservative parameters...")
# Calculate class ratio for handling imbalance
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    scale_pos_weight=ratio,  # Handle the class imbalance
    n_estimators=50,         # Reduced to prevent overfitting
    max_depth=3,             # Shallow trees to prevent overfitting
    learning_rate=0.05,      # Lower learning rate for stability
    subsample=0.8,           # Subsampling to prevent overfitting
    colsample_bytree=0.8,    # Feature subsampling to prevent overfitting
    min_child_weight=5,      # Higher min_child_weight to prevent overfitting
    reg_alpha=0.1,           # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    random_state=42
)
model.fit(X_train, y_train)

# 9. Evaluate
print("Evaluating model...")
preds = model.predict(X_test)
preds_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class

accuracy = accuracy_score(y_test, preds)
auc_score = roc_auc_score(y_test, preds_proba)
precision, recall, f1, support = precision_recall_fscore_support(y_test, preds, average='binary')

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"AUC Score: {auc_score:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print("\nDetailed Report:")
print(classification_report(y_test, preds))

# Confusion Matrix
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Significant', 'Significant'], 
            yticklabels=['Not Significant', 'Significant'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# 10. Province Analysis
print("\nAnalyzing significant earthquakes by province...")
# Since the dataset doesn't have explicit province names, we'll analyze by regions
region_counts = df.groupby(['Region', 'is_significant']).size().unstack(fill_value=0)
region_counts['Total'] = region_counts.sum(axis=1)
region_counts['Significant_Rate'] = region_counts[1] / region_counts['Total'] * 100
region_counts = region_counts.sort_values('Significant_Rate', ascending=False)

print("Regional Analysis:")
print(region_counts)

# 11. Save Model and Processed Data
print("Saving model and processed data...")
joblib.dump(model, 'earthquake_model_realistic.pkl')
df.to_csv('processed_earthquake_data_realistic.csv', index=False)

# Save feature columns for later use
joblib.dump(feature_cols, 'feature_columns_realistic.pkl')

print("Files saved successfully.")
print(f"Model saved as 'earthquake_model_realistic.pkl'")
print(f"Processed data saved as 'processed_earthquake_data_realistic.csv'")
print(f"Feature columns saved as 'feature_columns_realistic.pkl'")

# Create summary report
summary_report = f"""
Earthquake Prediction Model Summary Report
=========================================

Model Performance:
- Accuracy: {accuracy * 100:.2f}%
- AUC Score: {auc_score:.2f}
- Precision: {precision:.2f}
- Recall: {recall:.2f}
- F1-Score: {f1:.2f}

Dataset Information:
- Total records: {len(df)}
- Training records: {len(X_train)}
- Testing records: {len(X_test)}
- Features used: {len(feature_cols)}
- Significant earthquakes (>=4.0): {y.sum()} ({y.mean()*100:.2f}%)
- Non-significant earthquakes: {len(y) - y.sum()} ({(1-y.mean())*100:.2f}%)

Top 5 Most Important Features:
{feature_importance.head(5).to_string(index=False)}

Regional Analysis (Significant Quake Rate):
{region_counts[['Significant_Rate']].to_string()}
"""

with open('earthquake_analysis_report_realistic.txt', 'w') as f:
    f.write(summary_report)

print("Summary report saved as 'earthquake_analysis_report_realistic.txt'")