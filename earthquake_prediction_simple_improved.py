import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(n_samples=1000):
    """Create realistic earthquake data for the Philippines region"""
    np.random.seed(42)
    
    # Generate data points with higher density in seismically active areas
    # Focus on the Philippines (110-130°E, 5-20°N)
    latitudes = []
    longitudes = []
    
    # Simulate clustering around fault lines (e.g., Manila Trench, Philippine Fault)
    for _ in range(n_samples):
        if np.random.random() < 0.6:  # 60% in high-risk areas
            # High-risk zones (near fault lines)
            lat = np.random.normal(15, 3)  # Centered around 15°N
            lon = np.random.normal(122, 2)  # Centered around 122°E
        else:
            # Lower risk areas
            lat = np.random.uniform(5, 20)
            lon = np.random.uniform(110, 130)
        
        # Ensure values are within bounds
        lat = np.clip(lat, 5, 20)
        lon = np.clip(lon, 110, 130)
        
        latitudes.append(lat)
        longitudes.append(lon)
    
    # Create depth based on location (shallow near trenches, deeper inland)
    depths = []
    for lat, lon in zip(latitudes, longitudes):
        # Shallow near trenches, deeper inland
        if 120 <= lon <= 125 and 12 <= lat <= 18:
            depth = np.random.exponential(15)  # Shallow focus
        else:
            depth = np.random.exponential(25)  # Deeper focus
        
        depths.append(min(depth, 700))  # Cap at 700km
    
    # Create magnitude with correlation to depth (shallow quakes often more damaging)
    magnitudes = []
    for depth in depths:
        # Shallower quakes more likely to have higher magnitude
        base_mag = 3.0 + np.random.exponential(0.8)
        depth_factor = 1.0 if depth < 70 else 0.8
        mag = base_mag * depth_factor
        magnitudes.append(np.clip(mag, 1.0, 8.0))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Depth_In_Km': depths,
        'Magnitude': magnitudes
    })
    
    # Sort by date to simulate time series
    df = df.sort_values(['Latitude', 'Longitude']).reset_index(drop=True)
    
    return df

def enhance_features(df):
    """Enhance features for better prediction"""
    df_enhanced = df.copy()
    
    # 1. Depth-Magnitude Ratio (shallow + strong = more significant)
    df_enhanced['depth_magnitude_ratio'] = df_enhanced['Depth_In_Km'] / (df_enhanced['Magnitude'] + 1)
    
    # 2. Magnitude Squared (non-linear relationship)
    df_enhanced['magnitude_squared'] = df_enhanced['Magnitude'] ** 2
    
    # 3. Normalized Depth
    df_enhanced['depth_normalized'] = df_enhanced['Depth_In_Km'] / df_enhanced['Depth_In_Km'].max()
    
    # 4. Spatial interaction features
    df_enhanced['lat_lon_interaction'] = df_enhanced['Latitude'] * df_enhanced['Longitude']
    
    # 5. Distance from reference point (Manila)
    manila_lat, manila_lon = 14.5995, 120.9842
    df_enhanced['distance_from_manila'] = np.sqrt(
        (df_enhanced['Latitude'] - manila_lat)**2 + 
        (df_enhanced['Longitude'] - manila_lon)**2
    )
    
    return df_enhanced

def perform_clustering(df):
    """Perform DBSCAN clustering to identify hotspots"""
    coords = df[['Latitude', 'Longitude']].values
    
    # Use tighter clustering parameters for better regional specificity
    db = DBSCAN(eps=0.05, min_samples=5).fit(coords)  # Tighter clusters
    df['cluster_id'] = db.labels_
    
    # Add cluster statistics
    cluster_stats = df.groupby('cluster_id').agg({
        'Magnitude': ['mean', 'std', 'count'],
        'Depth_In_Km': ['mean', 'std']
    }).fillna(0)
    
    # Flatten column names
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns]
    cluster_stats = cluster_stats.add_prefix('cluster_')
    
    # Merge cluster stats back to main dataframe
    df = df.merge(cluster_stats, left_on='cluster_id', right_index=True, how='left')
    
    return df

def main():
    print("🚀 Earthquake Prediction Model (Enhanced)")
    print("=" * 50)
    
    # 1. Load or Create Data
    print("📊 Creating sample earthquake data...")
    df = create_sample_data(10000)  # Larger dataset for better training
    
    # 2. Enhance Features
    print("⚙️  Enhancing features...")
    df = enhance_features(df)
    
    # 3. Perform Clustering
    print("🗺️  Performing DBSCAN clustering...")
    df = perform_clustering(df)
    
    # 4. Create Target Variable
    df['is_significant'] = (df['Magnitude'] >= 4.0).astype(int)
    
    # 5. Prepare Features
    feature_columns = [
        'Latitude', 'Longitude', 'Depth_In_Km',
        'depth_magnitude_ratio', 'magnitude_squared', 
        'depth_normalized', 'lat_lon_interaction',
        'distance_from_manila', 'cluster_id',
        'cluster_Magnitude_mean', 'cluster_Magnitude_std', 'cluster_Magnitude_count',
        'cluster_Depth_In_Km_mean', 'cluster_Depth_In_Km_std'
    ]
    
    # Ensure all required columns exist
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features]
    y = df['is_significant']
    
    # Fill any remaining NaN values
    X = X.fillna(X.mean())
    
    # 6. Split Data (Stratified)
    print("📈 Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Significant quakes in test set: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}%)")
    
    # 7. Train Model with Optimized Parameters
    print("\n🤖 Training XGBoost model...")
    
    # Calculate class weight to handle imbalance
    neg_count = len(y_train) - y_train.sum()
    pos_count = y_train.sum()
    scale_pos_weight = neg_count / pos_count
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 8. Evaluate Model
    print("\n🎯 Model Evaluation:")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"AUC Score: {auc_score:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 9. Feature Importance
    print("\n📊 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # 10. Save Model and Processed Data
    print("\n💾 Saving model and processed data...")
    joblib.dump(model, 'earthquake_model_improved.pkl')
    df.to_csv('processed_earthquake_data_improved.csv', index=False)
    
    print("\n✅ Model training completed successfully!")
    print(f"   Model saved as: earthquake_model_improved.pkl")
    print(f"   Data saved as: processed_earthquake_data_improved.csv")
    print(f"   Final Accuracy: {accuracy * 100:.2f}%")
    
    return model, df

if __name__ == "__main__":
    model, data = main()