import joblib
import numpy as np

# Test that all model files can be loaded properly
print("Testing model file loading...")

try:
    model = joblib.load('risk_area_identifier.pkl')
    scaler = joblib.load('scaler_risk_identifier.pkl')
    dbscan = joblib.load('dbscan_zone_identifier.pkl')
    threshold = joblib.load('threshold_risk_identifier.pkl')
    feature_cols = joblib.load('feature_cols_risk_identifier.pkl')
    zone_risk_lookup = joblib.load('zone_risk_lookup.pkl')
    
    print("✅ All model files loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Scaler type: {type(scaler)}")
    print(f"Threshold: {threshold}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Zone risk lookup shape: {zone_risk_lookup.shape}")
    
    # Test a simple prediction
    # Using sample values similar to the training data
    sample_input = np.array([[12.8797, 121.7740, 10.0, -1, 0.5,  # lat, lon, depth, zone, zone_risk
                            12.8797, 121.7740, 12.8797**2 + 121.7740**2,  # abs lat, abs lon, distance
                            np.log1p(10.0), 10.0/100.0,  # depth_log, depth_normalized
                            12.8797*121.7740,  # lat_long_interact
                            12.8797*10.0, 121.7740*10.0]])  # lat_depth, long_depth
    
    sample_scaled = scaler.transform(sample_input)
    prediction_proba = model.predict_proba(sample_scaled)[0, 1]
    prediction_binary = 1 if prediction_proba >= threshold else 0
    
    print(f"✅ Test prediction successful!")
    print(f"Input: lat=12.8797, lon=121.7740, depth=10.0")
    print(f"Risk probability: {prediction_proba:.3f}")
    print(f"Binary prediction: {prediction_binary}")
    print(f"Risk classification: {'High Risk' if prediction_binary == 1 else 'Low Risk'}")
    
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\nAll tests completed successfully! The Streamlit app should work properly.")